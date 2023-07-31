
'''
    Main network that combines the color style extractor, the 
    colorization network and the discriminator.
'''

from .color_style_extractor import ColorStyleExtractor
from .colorization_network import ColorizationNet
from .discriminator import MultiScalePatchDiscriminator


# _____________________________________________________________________________
class MainNet():
    def __init__(self, device):
        
        color_style_extractor = ColorStyleExtractor()
        colorization_network = ColorizationNet()
        discriminator = MultiScalePatchDiscriminator()
        
        self.color_style_extractor = color_style_extractor
        self.colorization_network = colorization_network
        self.discriminator = discriminator

        self.device = device

    def to_device(self):
        self.color_style_extractor = self.color_style_extractor.to(self.device)
        self.colorization_network = self.colorization_network.to(self.device)
        self.discriminator = self.discriminator.to(self.device)

    def zero_grad(self):
        # Sets the gradients of all optimized Tensors to zero
        self.color_style_extractor.zero_grad(set_to_none=True)
        self.colorization_network.zero_grad(set_to_none=True)
        self.discriminator.zero_grad(set_to_none=True)

    def train(self):
        # Set the model in train mode.
        self.color_style_extractor.train()
        self.colorization_network.train()
        self.discriminator.train()

    def eval(self):
        # Set the model in evaluation mode.
        self.color_style_extractor.eval()
        self.colorization_network.eval()
        self.discriminator.eval()

    def state_dict(self):
        # Return a dictionary with the parameters of the model.
        return {
                'color_style_extractor': self.color_style_extractor.state_dict(),
                'colorization_network': self.colorization_network.state_dict(),
                'discriminator': self.discriminator.state_dict()
            }

    def load_state_dict(self, state_dict):
        # Load previous parameters using a dictionary with the parameters of the model
        self.color_style_extractor.load_state_dict(state_dict['color_style_extractor'])
        self.colorization_network.load_state_dict(state_dict['colorization_network'])
        self.discriminator.load_state_dict(state_dict['discriminator'])
