import segmentation_models_pytorch as smp
import torch.nn as nn

AVAILABLE_MODEL_ARCHITECTURE = ['unet', 'fpn', 'DeepLabV3Plus']

class ModelFactory(nn.Module):

    """A tool that construct pytorch segmentation model
    Parameters
    ----------
    architecture : str
        name of architecture. Defaults to 'unet'.
    encoder_name : str
        name of encoder. Defaults to 'mobilenet_v2'.
    class_num : int
        number of classes. Defaults to 10.
    in_channels : int
        size of input channels. Defaults to 3.
    activation : string
        the activation function. Defaults to 'softmax'.

    """

    def __init__(self, architecture = 'unet', encoder_name = 'mobilenet_v2', class_num = 10, in_channels = 3, activation=None):
        super(ModelFactory, self).__init__()

        if architecture in AVAILABLE_MODEL_ARCHITECTURE:
            self.architecture = architecture
        else:    
            raise NotImplementedError('{} has not been implemented, use model in {}'.format(self.architecture,AVAILABLE_MODEL_ARCHITECTURE))

        # unet
        if self.architecture == 'unet':
            self.model = smp.Unet(encoder_name=encoder_name, encoder_weights='imagenet', in_channels=in_channels, classes=class_num, activation = activation)
            
        # fpn
        elif self.architecture == 'fpn':
            self.model = smp.FPN(encoder_name=encoder_name, encoder_weights='imagenet', in_channels=in_channels, classes=class_num)

        # mobilenet_v2: optimal input shape (224 * 224)
        elif self.architecture == 'DeepLabV3Plus':
            self.model = smp.DeepLabV3Plus(encoder_name=encoder_name, encoder_weights='imagenet', in_channels=in_channels, classes=class_num)

    def forward(self, x):
        return self.model(x)