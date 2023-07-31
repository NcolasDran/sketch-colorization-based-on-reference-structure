

import logging
import numpy as np
import configparser
from torch.utils.data import DataLoader

from utils import Messages, get_device
from training.data_preprocessing import collate_func
from training.adversarial_training import adversarial
from training.losses import MultiScaleReconstructionLoss

# Class for displaying messages in terminal
m = Messages()

# Load config values
config = configparser.ConfigParser()
config.read('configs/training.cfg')
BATCH_SIZE = config.getint('Training', 'batch_size')
FIXED_SIZE_IMGS = config.getboolean('Training', 'fixed_size_imgs')
TRAIN_IMG_SIZE = config.getint('Training', 'train_img_size')
WORKERS_DATALOADER = config.getint('Commons', 'workers_dataloader')

device = get_device()

reconstruction_loss = MultiScaleReconstructionLoss()
reconstruction_loss.to(device)

# _____________________________________________________________________________
# Train one Epoch
def training_epoch(image_dataset, net, optimizer_net):

    # New DataLoader for each epoch
    imgs_data_loader = DataLoader(image_dataset, batch_size=BATCH_SIZE, 
                                  shuffle=True, num_workers=WORKERS_DATALOADER,
                                  drop_last=True, pin_memory=True,
                                  collate_fn=(lambda batch: collate_func(batch,
                                                                         FIXED_SIZE_IMGS,
                                                                         resize_batch=TRAIN_IMG_SIZE,
                                                                         train=True)))
    # Dictionary to store the train losses
    epoch_train_losses = {'rec_loss': [], 'adv_loss': [], 'total_loss': []}

    ###################
    #   Train Batch   #
    ###################
    logging.critical('Running epoch... \n')
    # Go through all the batches of the epoch
    for batch_ndx, img_batch in enumerate(imgs_data_loader, start=1):

        # Send image batch to device
        img_batch.to_device(device)

        # Set the gradients to zero
        optimizer_net.zero_grad(set_to_none=True)

        # Extract color style from the color sketch reference
        color_style = net.color_style_extractor(img_batch.color_sketch) 

        # Generate the colored sketches, based con the color style reference
        img_l1, img_l2, img_l3, img_l4 = net.colorization_network(img_batch.sketch,
                                                                  color_style)

        # Multi-scale reconstruction loss
        rec_loss = reconstruction_loss(img_batch, img_l1, img_l2, img_l3, img_l4)

        # Multi-scale adversarial loss
        adv_loss = adversarial(net.discriminator, img_batch, img_l1, img_l2,
                               img_l3, img_l4)

        # Sum of the reconstruction loss and the adversarial loss
        total_loss = rec_loss + adv_loss
        total_loss.backward()

        # Append the loss results
        epoch_train_losses['rec_loss'].append(rec_loss.item())
        epoch_train_losses['adv_loss'].append(adv_loss.item())
        epoch_train_losses['total_loss'].append(total_loss.item())

        optimizer_net.step()

    # Return the mean of the losses for the epoch
    return {k: np.mean(v) for k, v in epoch_train_losses.items()}
