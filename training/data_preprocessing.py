
import os
import torch
import random
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image, ImageReadMode
import torchvision.transforms as T
import torchvision.transforms.functional as F


'''
    Custom values of mean and standard deviation calculated
    using the class CustomMeanStandardDeviation
    (this values are used in VGG16PerceptualLoss)
'''
# '''
# For dataset 'Anime Sketch Colorization Pair'
TRAIN_MEAN_COLOR = torch.tensor([0.8752, 0.8437, 0.8398])
TRAIN_STD_COLOR = torch.tensor([0.2417, 0.2681, 0.2663])
# '''

'''
# For dataset 'OPM Colorization' (fine-tune dataset)
TRAIN_MEAN_COLOR = torch.tensor([-0.3391, -0.3608, -0.3653])
TRAIN_STD_COLOR = torch.tensor([0.7500, 0.7308, 0.7218])
# '''


# Values to normalize image (technically, it would be Scale)
mean = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32)
std = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32)

NORMALIZE =	T.Normalize(mean.tolist(), std.tolist())

INVERSE_NORMALIZE = T.Normalize((-mean / std).tolist(), 
                                (1.0 / std).tolist())


# ____________________________________________________________________________
#   Some functions obtained and modified from 
#   https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
    '.tif', '.TIF', '.tiff', '.TIFF',
]

# __________________________
def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

# ____________________________________________________
def get_img_paths(dir, max_dataset_size=float('inf')):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir
    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)
    return sorted(images[:min(max_dataset_size, len(images))])


# ____________________________________________________________________________
class CustomImgDataset(Dataset):
    '''
        Custom Dataset class to load and transform the images, returning the
        elements individually
    '''

    def __init__(self, imgs_dir, fixed_size_imgs, split=True, random_flip=False):

        img_paths = get_img_paths(imgs_dir)
        if split:
            # Each image is a pair of sketch and colored sketch
            self.total_imgs = len(img_paths)
            self.img_paths = img_paths
        else:
            # If the sketch and the color reference are not together in 
            # the same image, then the name of both images must be 
            # "[something]_color" and "[something]_sketch".
            img_paths.sort()
            # Reorder the data as [[something_color, something_sketch], [..., ...]]
            self.img_paths = [[img_paths[i], img_paths[i+1]] for i in
                              range(0,len(img_paths),2)]
            self.total_imgs = len(self.img_paths)

        self.split = split
        self.fixed_size_imgs = fixed_size_imgs
        self.random_flip = random_flip
    
    # ________________________________________________________________________
    def get_ltrb(self, diff):
        '''
            For padding, get left, top, right and bottom.
        '''
        if diff > 0:
            # Pad width
            if diff%2:
                left = diff//2
                right = diff//2 + 1
            else:
                left, right = diff//2, diff//2
            top, bottom = 0, 0
        else:
            diff*=-1
            # Pad height
            left, right = 0, 0
            if diff%2:
                top = diff//2
                bottom = diff//2 + 1
            else:
                top, bottom = diff//2, diff//2
        return left, top, right, bottom

    # ________________________________________________________________________
    def pad_imgs(self, color_sketch, sketch):
        
        if sketch.shape == color_sketch.shape:
            diff = color_sketch.shape[1] - color_sketch.shape[2]
            left, top, right, bottom = self.get_ltrb(diff)
            color_sketch = F.pad(color_sketch, [left, top, right, bottom])
            sketch = F.pad(sketch, [left, top, right, bottom])
        else:
            diff_color_sketch = color_sketch.shape[1] - color_sketch.shape[2]
            diff_sketch = sketch.shape[1] - sketch.shape[2]
            left_cs, top_cs, right_cs, bottom_cs = self.get_ltrb(diff_color_sketch)
            left_s, top_s, right_s, bottom_s = self.get_ltrb(diff_sketch)
            color_sketch = F.pad(color_sketch, [left_cs, top_cs, right_cs, bottom_cs])
            sketch = F.pad(sketch, [left_s, top_s, right_s, bottom_s])
        
        return color_sketch, sketch

    # ________________________________________________________________________
    def __len__(self):
        return self.total_imgs

    # ________________________________________________________________________
    def __getitem__(self, idx):
        '''
            Method that returns two images, the sketch and the colored sketch
        '''
        img_path = self.img_paths[idx]
        if self.split:
            # Each image has the sketch and the color reference
            img = read_image(img_path)
            if self.fixed_size_imgs:
                # Fixed values from the 'Anime Sketch Colorization Pair' dataset
                height = 512
                width = 1024
            else:
                _, height, width = img.shape
            # The colored sketch takes the first half of the image
            color_sketch = T.functional.crop(img, 0, 0, height, width//2)
            # The sketch takes the second half of the image
            sketch = T.functional.crop(img, 0, width//2, height, width//2)
        else:
            img_color, img_sketch = img_path
            color_sketch = read_image(img_color, mode=ImageReadMode.RGB)
            sketch = read_image(img_sketch, mode=ImageReadMode.RGB)
            if not self.fixed_size_imgs:
                # Add padding to make square images
                color_sketch, sketch = self.pad_imgs(color_sketch, sketch)

        # If images have more than three channels, keep only the first three
        if color_sketch.shape[0] > 3:
            color_sketch = color_sketch[:3,:,:]
        if sketch.shape[0] > 3:
            sketch = sketch[:3,:,:]

        if self.random_flip:
            if random.random() > 0.5:
                color_sketch = F.hflip(color_sketch)
                sketch = F.hflip(sketch)

        return color_sketch, sketch

# ______________________________________________
def get_transformation_fixed(resize, norm=None):
    '''
        Get the transformations for fixed size images
    '''
    img_transformations = []
    # From uint8 to float
    img_transformations.append(T.ConvertImageDtype(torch.float))
    # Resize images for training
    img_transformations.append(T.Resize(resize, antialias=True))
    # Normalize image
    if norm is not None:
        img_transformations.append(NORMALIZE)
    return T.Compose(img_transformations)

# _____________________________________________________
def transform_batch_imgs(batch_img, resize, norm=None):
    '''
        Apply transformations to the batch of images
    '''
    # Resize images for training
    batch_img_res = [F.resize(img, resize, antialias=True) for img in
                     batch_img]
    img_transformations = []
    # From uint8 to float
    img_transformations.append(T.ConvertImageDtype(torch.float))
    # Normalize image
    if norm is not None:
        img_transformations.append(NORMALIZE)
    # Apply transformations to all images
    img_transformations = T.Compose(img_transformations)
    return img_transformations(torch.stack(batch_img_res, dim=0))


# ____________________________________________________________________________
class CustomImageBatch:
    '''
        Custom class to process a batch of images
    '''
    def __init__(self, img_batch, fixed_size, resize_batch, train, diff_sketch):
        
        self.train = train
        self.diff_sketch = diff_sketch

        color_sketch, sketch = zip(*img_batch)

        if fixed_size: # All the images have the same size
            img_transformations_color = get_transformation_fixed(resize_batch, norm='color')
            img_transformations_sketch = get_transformation_fixed(resize_batch, norm='sketch')
            self.color_sketch = img_transformations_color(torch.stack(list(color_sketch), dim=0))
            self.sketch = img_transformations_sketch(torch.stack(list(sketch), dim=0))
        else: # The images have different sizes
            self.color_sketch = transform_batch_imgs(color_sketch, resize_batch, norm='color')
            self.sketch = transform_batch_imgs(sketch, resize_batch, norm='sketch')

        if self.train:
            # Get output resolutions for each level generated by the colorization network
            original_res = tuple(self.sketch.shape[2:4])
            res_l1 = tuple(res//8 for res in original_res)
            res_l2 = tuple(res//4 for res in original_res)
            res_l3 = tuple(res//2 for res in original_res)
            # Resize the color sketch to the resolution of each level
            self.l1_color_sketch = T.Resize(res_l1, antialias=True)(self.color_sketch)
            self.l2_color_sketch = T.Resize(res_l2, antialias=True)(self.color_sketch)
            self.l3_color_sketch = T.Resize(res_l3, antialias=True)(self.color_sketch)

        if self.diff_sketch:
            # For validation/testing, assign a different colored sketch to the sketch
            rearrange_idx = torch.arange(len(img_batch))             
            np.random.shuffle(rearrange_idx.numpy())
            self.color_sketch = self.color_sketch[rearrange_idx]

    # ___________________________________________
    # Custom memory pinning method on custom type
    def pin_memory(self):
        self.color_sketch = self.color_sketch.pin_memory()
        self.sketch = self.sketch.pin_memory()
        if self.train:
            self.l1_color_sketch = self.l1_color_sketch.pin_memory()
            self.l2_color_sketch = self.l2_color_sketch.pin_memory()
            self.l3_color_sketch = self.l3_color_sketch.pin_memory()
        return self

    # _____________________________________
    # Custom method to send batch to device
    def to_device(self, device):
        self.color_sketch = self.color_sketch.to(device, non_blocking=True)
        self.sketch = self.sketch.to(device, non_blocking=True)
        if self.train:
            self.l1_color_sketch = self.l1_color_sketch.to(device, non_blocking=True)
            self.l2_color_sketch = self.l2_color_sketch.to(device, non_blocking=True)
            self.l3_color_sketch = self.l3_color_sketch.to(device, non_blocking=True)

# ____________________________________________________________________________
# Returns custom (processed) batch of images
def collate_func(img_batch, fixed_size, resize_batch, train=False, diff_sketch=False):
    return CustomImageBatch(img_batch, fixed_size, resize_batch, train, diff_sketch)



# _____________________________________________________________________________
class CustomMeanStandardDeviation:
    '''
        Calculate mean and std for the train dataset
        (used in VGG16PerceptualLoss)

        Code obtained and modified from
        https://github.com/Nikronic/CoarseNet/blob/master/utils/preprocess.py#L142-L200
        
        mean, std = CustomMeanStandardDeviation()(image_dataset, batch_size=1)
    '''
    def __init__(self):
        pass

    def __call__(self, image_dataset, batch_size=1):

        loader = DataLoader(image_dataset, batch_size=batch_size, shuffle=False, 
                            num_workers=1, pin_memory=0, collate_fn=(lambda batch: collate_func(batch,
                                                                                                False, 
                                                                                                resize_batch=256,
                                                                                                train=True)))
        print('Total',len(loader))
        # 'weak' method
        if batch_size > 1:
            mean_color = 0.
            std_color = 0.
            mean_sketch = 0.
            std_sketch = 0.
            nb_color_samples = 0.
            nb_sketch_samples = 0.
            # data = data.sketch
            for batch_ndx, img_batch in enumerate(loader, start=1):
                print(batch_ndx)
                # Color img
                data_color = img_batch.color_sketch
                batch_color_samples = data_color.size(0)
                data_color = data_color.view(batch_color_samples, data_color.size(1), -1)
                mean_color += data_color.mean(2).sum(0)
                std_color += data_color.std(2).sum(0)
                nb_color_samples += batch_color_samples
                # Sketch img
                data_sketch = img_batch.sketch
                batch_sketch_samples = data_sketch.size(0)
                data_sketch = data_sketch.view(batch_sketch_samples, data_sketch.size(1), -1)
                mean_sketch += data_sketch.mean(2).sum(0)
                std_sketch += data_sketch.std(2).sum(0)
                nb_sketch_samples += batch_sketch_samples

            mean_color /= nb_color_samples
            std_color /= nb_color_samples
            mean_sketch /= nb_sketch_samples
            std_sketch /= nb_sketch_samples

            return mean_color, std_color, mean_sketch, std_sketch

        # 'strong' method
        else:
            cnt_color = 0
            fst_moment_color = torch.empty(3)
            snd_moment_color = torch.empty(3)
            cnt_sketch = 0
            fst_moment_sketch = torch.empty(3)
            snd_moment_sketch = torch.empty(3)

            for batch_ndx, img_batch in enumerate(loader, start=1):
                if batch_ndx%50 == 0:
                    print(batch_ndx)
                # Color img
                data_color = img_batch.color_sketch
                b, c, h, w = data_color.shape
                nb_pixels_color = b * h * w
                sum_ = torch.sum(data_color, dim=[0, 2, 3])
                sum_of_square = torch.sum(data_color ** 2, dim=[0, 2, 3])
                fst_moment_color = (cnt_color * fst_moment_color + sum_) / (cnt_color + nb_pixels_color)
                snd_moment_color = (cnt_color * snd_moment_color + sum_of_square) / (cnt_color + nb_pixels_color)
                cnt_color += nb_pixels_color
                # Sketch img
                data_sketch = img_batch.sketch
                b, c, h, w = data_sketch.shape
                nb_pixels_sketch = b * h * w
                sum_ = torch.sum(data_sketch, dim=[0, 2, 3])
                sum_of_square = torch.sum(data_sketch ** 2, dim=[0, 2, 3])
                fst_moment_sketch = (cnt_sketch * fst_moment_sketch + sum_) / (cnt_sketch + nb_pixels_sketch)
                snd_moment_sketch = (cnt_sketch * snd_moment_sketch + sum_of_square) / (cnt_sketch + nb_pixels_sketch)
                cnt_sketch += nb_pixels_sketch

            return fst_moment_color, torch.sqrt(snd_moment_color - fst_moment_color ** 2), fst_moment_sketch, torch.sqrt(snd_moment_sketch - fst_moment_sketch ** 2)
