
import time
import torch
import logging
import configparser

from training.data_preprocessing import CustomImgDataset
from training.train_epoch import training_epoch
from training.validation import validation_loop

from utils import (
    Messages,
    save_to_ckpt,
    load_from_ckpt,
)

# Class for displaying messages in terminal
m = Messages()

# Load config values
config = configparser.ConfigParser()
config.read('configs/training.cfg')

#---------------------
#   Config Training
#---------------------
MAX_EPOCH = config.getint('Training', 'max_epoch')
LEARNING_RATE = config.getfloat('Training', 'learning_rate')
SAVE_CHECKPOINT = config.getboolean('Training', 'save_checkpoint')
DATASET_NAME = config.get('Training', 'dataset_name')
SPLIT_BOOL = config.getboolean('Training', 'split_img')
RANDOM_TRANSFORM = config.getboolean('Training', 'random_transform')
FIXED_SIZE_IMGS = config.getboolean('Training', 'fixed_size_imgs')


# _____________________________________________________________________________
# Execution of the model training  
def training_loop(net):

    logging.critical(m.train)

    # Load network checkpoint
    start_epoch, epoch_losses, val_results = load_from_ckpt(net)
    net.to_device()

    if len(epoch_losses) == 0:
        epoch_losses = {'rec_loss': [], 'adv_loss': [], 'total_loss': []}
        val_results = {'PSNR':[], 'SSIM':[], 'FID':[]}

    # Optimizer with network parameters
    optimizer_net = torch.optim.Adam(
                [
                    {'params': net.color_style_extractor.parameters()},
                    {'params': net.colorization_network.parameters()},
                    {'params': net.discriminator.parameters()},
                ], lr=LEARNING_RATE
            )
    
    # Learning rate scheduler
    lr_scheduler_net = torch.optim.lr_scheduler.PolynomialLR(
                        optimizer=optimizer_net, 
                        total_iters=MAX_EPOCH, power=0.9
                    )

    # Load checkpoint for optimizer and scheduler
    load_from_ckpt(optimizer_net=optimizer_net,
                   lr_scheduler_net=lr_scheduler_net)

    # Load train data
    path_data = 'data/' + DATASET_NAME + '/train/'
    image_dataset = CustomImgDataset(path_data, FIXED_SIZE_IMGS, split=SPLIT_BOOL,
                                     random_flip=RANDOM_TRANSFORM)

    # _________________________________________________________________________
    ####################
    #   Train Epochs   #
    ####################

    #  Go through the epochs of training
    for epoch in range(start_epoch, MAX_EPOCH + 1):
        
        start_epoch_time = time.time()
        logging.critical(m.epoch_beginning(epoch))

        # Model gradients to zero
        net.zero_grad()
        # Model in train mode
        net.train()

        #---------------------
        #   Train one Epoch
        #---------------------
        batch_losses = training_epoch(image_dataset, net,
                                      optimizer_net)

        lr_scheduler_net.step()

        # Append losses for epoch
        for k,v in batch_losses.items():
            epoch_losses[k].append(v)

        # Display in terminal the epoch results
        logging.critical(m.epoch_results(str(epoch), epoch_losses,
                         round(time.time() - start_epoch_time)))

        #--------------------
        #   Validation loop
        #--------------------
        start_val_time = time.time()

        # Execute validation and return metrics (SNR, SSIM and FID)
        val_metrics_results = validation_loop(net, epoch)
        # Append validation results
        for k,v in val_metrics_results.items():
            val_results[k].append(v)

        # Display in terminal the validation results
        logging.critical(m.val_results(str(epoch), val_results,
                         round(time.time() - start_val_time)))

        # Save checkpoint
        if SAVE_CHECKPOINT: 
            save_to_ckpt(epoch, net, optimizer_net, lr_scheduler_net,
                         epoch_losses, val_results)
