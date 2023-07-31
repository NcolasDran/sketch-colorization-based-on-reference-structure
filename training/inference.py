
import torch
from torch.utils.data import DataLoader

from utils import load_from_ckpt
from training.data_preprocessing import CustomImgDataset, collate_func

# ____________________________________________________________________________
def inference_loop(net, device):

    # Load checkpoint, only for the model
    load_from_ckpt(net)
    net.to_device()
    # Set the model in evaluation mode.
    net.eval()

    # -----------------------------------
    batch_size = 1
    fixed_size = False
    img_split = False
    img_resize = 256 # (or None)
    data_path = 'data/inference/'
    # -----------------------------------

    # Load data and generate a DataLoader
    img_dataset = CustomImgDataset(data_path, fixed_size, split=img_split)
    img_data_loader = DataLoader(img_dataset, batch_size=batch_size, pin_memory=True,
                                 collate_fn=(lambda batch: collate_func(batch,
                                                                        fixed_size,
                                                                        resize_batch=img_resize,
                                                                        diff_sketch=True)))
    ################
    #   Inference  #
    ################
    # Go through all the batches
    for batch_ndx, img_batch in enumerate(img_data_loader, start=1):

        # Send batch to device
        img_batch.to_device(device)
        # Disable gradient calculation
        with torch.no_grad():
            # Extract color style from the color sketch reference
            color_style = net.color_style_extractor(img_batch.color_sketch)

            # Generate the colored sketches, based con the color style reference
            out_img = net.colorization_network(img_batch.sketch, color_style, multi_scale=False)

        ## do something with the output colored sketch ##
