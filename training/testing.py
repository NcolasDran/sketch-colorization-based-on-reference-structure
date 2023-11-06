
'''
    The 'PyTorch Image Quality' (PIQ) library is used for the evaluation metrics
    https://github.com/photosynthesis-team/piq
    $ pip install piq
'''

import torch
import logging
import configparser
import numpy as np
from torch.utils.data import DataLoader
from piq import ssim, psnr, FID  # PyTorch Image Quality

from training.data_preprocessing import CustomImgDataset, collate_func, INVERSE_NORMALIZE
from utils import Messages, load_from_ckpt

# Class for displaying messages in terminal
m = Messages()

# Load config values
config = configparser.ConfigParser()
config.read('configs/training.cfg')

#----------------------
#   Config testing
#----------------------
DATASET_NAME = config.get('Testing', 'dataset_name')
BATCH_SIZE = config.getint('Testing', 'batch_size')
SPLIT_BOOL = config.getboolean('Testing', 'split_img')
WORKERS_DATALOADER = config.getint('Commons', 'workers_dataloader')
FIXED_SIZE_IMGS = config.getboolean('Testing', 'fixed_size_imgs')
TEST_IMG_SIZE = config.getint('Testing', 'test_img_size')

if DATASET_NAME == 'anime_sketch_colorization_pair':
    # This dataset only has validation data, so this images are 
    # used for the testing fase.
    PATH_DATA = 'data/' + DATASET_NAME + '/val/'
else:
    PATH_DATA = 'data/' + DATASET_NAME + '/test/'

# _____________________________________________________________________________
class CustomDataFID(torch.utils.data.Dataset):
    '''
        Custom Dataset needed to calculate features for FID metric.
        https://github.com/photosynthesis-team/piq/issues/241#issuecomment-809173818
    '''

    def __init__(self, batch_img):
        super(CustomDataFID, self).__init__()
        self.batch_img = batch_img

    def __getitem__(self, index):
        return {
            'images': self.batch_img[index],
        }

    def __len__(self):
        return len(self.batch_img)

# _____________________________________________________________________________
def testing_loop(net, device):

    logging.critical(m.test)
    # Load checkpoint
    load_from_ckpt(net)
    net.to_device()
    # Model in evaluation mode
    net.eval()

    #-------------------------
    #   Load testing data
    #-------------------------
    image_dataset = CustomImgDataset(PATH_DATA, FIXED_SIZE_IMGS, split=SPLIT_BOOL, 
                                     random_flip=False)

    imgs_data_loader = DataLoader(image_dataset, batch_size=BATCH_SIZE, 
                                  shuffle=False, num_workers=WORKERS_DATALOADER,
                                  drop_last=True, pin_memory=True,
                                  collate_fn=(lambda batch: collate_func(batch,
                                                                         FIXED_SIZE_IMGS,
                                                                         resize_batch=TEST_IMG_SIZE,
                                                                         diff_sketch=False)))

    test_metrics = {'PSNR':[], 'SSIM':[], 'FID':[]}

    fid_metric = FID()

    logging.critical('Running testing... \n')
    # Go through all testing batches
    for batch_ndx, img_batch in enumerate(imgs_data_loader, start=1):

         # Send image batch to device
        img_batch.to_device(device)

        # Disable gradient calculation
        with torch.no_grad():
            # Extract color style from the color sketch reference
            color_style = net.color_style_extractor(img_batch.color_sketch) 

            # Generate the colored sketch, based con the color style reference
            out_img = net.colorization_network(img_batch.sketch, color_style,
                                               multi_scale=False)

        # ---------------- Testing metrics ---------------- #

        inv_out_img = INVERSE_NORMALIZE(out_img)
        inv_color_sketch = INVERSE_NORMALIZE(img_batch.color_sketch)

        # Peak signal-to-noise ratio 
        psnr_index = psnr(inv_out_img, inv_color_sketch, data_range=1.)
        # Structural similarity index 
        ssim_index = ssim(inv_out_img, inv_color_sketch, data_range=1.)
        
        # Fréchet inception distance
        # https://github.com/photosynthesis-team/piq/issues/241#issuecomment-809173818
        data_out_img = CustomDataFID(inv_out_img)
        data_color_sketch = CustomDataFID(inv_color_sketch)
        out_img_dl = DataLoader(data_out_img, batch_size=1, shuffle=False)
        color_sketch_dl = DataLoader(data_color_sketch, batch_size=1, shuffle=False)
    
        out_img_feats = fid_metric.compute_feats(out_img_dl)
        color_sketch_feats = fid_metric.compute_feats(color_sketch_dl)

        fid = fid_metric.compute_metric(out_img_feats.to(device),
                                        color_sketch_feats.to(device))

        test_metrics['PSNR'].append(psnr_index.item())
        test_metrics['SSIM'].append(ssim_index.item())
        test_metrics['FID'].append(fid.item())
    
    # Return the mean of metrics
    return {k: np.mean(v) for k, v in test_metrics.items()}
