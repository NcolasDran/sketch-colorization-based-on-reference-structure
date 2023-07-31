

import os
import datetime
import torch
import logging
import configparser
import torchvision.transforms as T

from training.data_preprocessing import INVERSE_NORMALIZE


# Load config values
config = configparser.ConfigParser()
config.read('configs/training.cfg')
# Learning rate
LR = config.get('Training', 'learning_rate').replace('-', '')
FINE_TUNE = config.getboolean('Training', 'fine_tune')

# _____________________________________________________________________________
def get_time():
    # Return a string with the current time
    return datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

# _____________________________________________________________________________
def get_device():
    # Set the device used for training ('cuda:0' is for GPU).
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # device = torch.device(type='cpu') # Set device as CPU only
    return device

# _____________________________________________________________________________
def set_logging(set_level):
    '''
    Set a Logging Level:
        - logging.debug
        - logging.info
        - logging.warning
        - logging.error
        - logging.critical
    '''

    if set_level == 'debug':
        logging.basicConfig(format='%(message)s', level=logging.DEBUG)
    elif set_level == 'info':
        logging.basicConfig(format='%(message)s', level=logging.INFO)
    elif set_level == 'warning':
        logging.basicConfig(format='%(message)s', level=logging.WARNING)
    elif set_level == 'error':
        logging.basicConfig(format='%(message)s', level=logging.ERROR)
    elif set_level == 'critical':
        logging.basicConfig(format='%(message)s', level=logging.CRITICAL)


# _____________________________________________________________________________
def get_ckpt_path(epoch=0):
    ''' 
        Return the path for the checkpoints folder, and a name for a current
        checkpoint.
    '''

    # Path of the checkpoints folder
    ckpt_dir = os.path.join('data', 'checkpoints')
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    # Current time
    time = datetime.datetime.now().strftime('%Y_%m_%d__%H_%M_%S')

    if FINE_TUNE:
        ft = 'fine-tune_'
    else:
        ft = ''

    ckpt_name = 'Epoch_%d--' + ft + 'lr_' + LR + '-' + time + '.pt.tar'
    return ckpt_dir, ckpt_name % (epoch)

# _____________________________________________________________________________
def save_to_ckpt(epoch, net, optimizer_net, lr_scheduler_net, epoch_losses, 
                 val_metric_results):
    '''
        For each Epoch, save the parameters of the model, the optimizers, the 
        scheduler and the results of the losses.
    '''

    ckpt_dir, ckpt_name = get_ckpt_path(epoch)
    logging.critical('\n Creating checkpoint...')
    torch.save({
        'epoch': epoch,
        'epoch_losses': epoch_losses,
        'val_results': val_metric_results,
        'net_state_dict': net.state_dict(),
        'optimizer_net_state_dict': optimizer_net.state_dict(),
        'lr_scheduler_net': lr_scheduler_net.state_dict(),
    }, os.path.join(ckpt_dir, ckpt_name))
    logging.critical(" Checkpoint created in '" + ckpt_dir + "'\n")


# _____________________________________________________________________________
def load_from_ckpt(net=None, optimizer_net=None, lr_scheduler_net=None):
    '''
        Load the parameters of the last checkpoint to the model, the optimizer
        or the scheduler.
    '''

    start_epoch = 1
    epoch_losses = {}
    val_metric_results = {}

    ckpt_dir, _ = get_ckpt_path()
    # The names of all the checkpoints
    ckpts = [ckpt for ckpt in os.listdir(ckpt_dir) if '.pt.tar' in ckpt]
    try:
        if len(ckpts) > 0:
            last_epoch = [0, '-']
            # Iterate the names
            for ckpt in ckpts:
                # Number of epoch
                ep = int(ckpt.split('--')[0].split('_')[1])
                # Get last checkpoint
                if ep > last_epoch[0]:
                    last_epoch = [ep, ckpt]

            ckpt_path = os.path.join(ckpt_dir, last_epoch[1])
            if os.path.isfile(ckpt_path):
                if net is not None: logging.critical("\n Loading checkpoint from '" + ckpt_path + "'...")
                # Load parameters in checkpoint
                state_dict = torch.load(ckpt_path)

                # Load state dict of network
                if net is not None:
                    if FINE_TUNE:
                        if 'fine-tune' in ckpt_path:
                            # Add new block in color extractor, and then load checkpoint
                            net.color_style_extractor.add_block5()
                            net.load_state_dict(state_dict['net_state_dict'])
                        else:
                            # Load checkpoint, and then add new block in color extractor
                            net.load_state_dict(state_dict['net_state_dict'])
                            net.color_style_extractor.add_block5()
                    else:
                        net.load_state_dict(state_dict['net_state_dict'])

                # Load state dict of optimizer and scheduler
                if (not FINE_TUNE) or (FINE_TUNE and 'fine-tune' in ckpt_path):
                    start_epoch = state_dict['epoch'] + 1
                    epoch_losses = state_dict['epoch_losses']
                    val_metric_results = state_dict['val_results']
                    if optimizer_net is not None:
                        optimizer_net.load_state_dict(state_dict['optimizer_net_state_dict'])
                    if lr_scheduler_net is not None:
                        lr_scheduler_net.load_state_dict(state_dict['lr_scheduler_net'])

                if net is not None: logging.critical(' Checkpoint loaded\n')
        else:
            logging.critical('\n No checkpoint found\n')
    except:
        logging.critical(' Error found, no checkpoint loaded\n')
    return start_epoch, epoch_losses, val_metric_results


# _____________________________________________________________________________
class Messages():
    '''
        Class for displaying messages in terminal
    '''
    def __init__(self):
        self.batch_beginning = ('_'*30 + '\n' + 'Batch %d/%d\n')
        self.debug = '    > %s'
        self.train = ('\n'+'#'*22+'\n#'+' '*6+'TRAINING'+' '*6+'#\n'+'#'*22+'\n')
        self.val = ('\n '+'-'*20+'\n|'+' '*5+'VALIDATION'+' '*5+'|\n '+'-'*20+'\n')
        self.test = ('\n'+'#'*22+'\n#'+' '*6+'TESTING'+' '*6+'#\n'+'#'*22+'\n')

    #________________________________________________________
    # Batch loss results
    def batch(self, batch_ndx, n_batches, epoch_train_losses):
        log = (' | Batch '+batch_ndx+'/'+n_batches + ' |')
        log += ('  Rec loss: '+str(round(epoch_train_losses['rec_loss'][-1], 4))+' ')
        log += ('ADV loss: '+str(round(epoch_train_losses['adv_loss'][-1], 4))+' ')
        return log
    #________________________________________________________
    # Start of Epoch
    def epoch_beginning(self, epoch):
        return(('\n ' + '-'*13 + '\n' + '|   Epoch ' +
                str(epoch) + '   |\n ' + '-'*13 +'\n\n' +
                'Start: ' + get_time()))
    #________________________________________________________
    # Epoch loss results
    def epoch_results(self, epoch, epoch_losses, total_time):
        log = '\n'+'='*32 + '\n' + get_time() + ' | Epoch ' + epoch + '\n\n'
        for k,v in epoch_losses.items():
            log += (k + ': ' + str(round(v[-1], 3)) + '\n')
        log += '='*32 + '\n --- ' + str(total_time) + ' seconds total ---\n'
        return log
    #________________________________________________________
    # Validation Accuracy
    def val_results(self, epoch, val_metrics, total_time):
        log = '\n'+'='*32 + '\n' + get_time() + ' | Epoch ' + epoch + '\n\n'
        for k, v in val_metrics.items():
            log += (k + ': ' + str(round(v[-1], 3)) + '\n')
        log += '='*32 + '\n --- ' + str(total_time) + ' seconds total ---\n'
        return log
    #________________________________________________________
    # Current device
    def device(self, device):
        return('\n Using device: ' + device.__str__() + 
              ((' ('+ torch.cuda.get_device_name(0) +
              ')\n') if device.type == 'cuda' else '\n'))
