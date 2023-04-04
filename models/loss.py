from torch import nn
import torch.nn.functional as F
from torch.nn import (CrossEntropyLoss, BCELoss)
AVAILABLE_LOSS = ['binary_crossentropy', 'crossentropy', 'diceloss']


class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()
    
    def forward(self, inputs, targets, smooth=1):
        #
        inputs = F.sigmoid(inputs)
        #flatten
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        #element_wise production to get intersection score
        intersection = (inputs*targets).sum()
        dice_score = (2*intersection + smooth) / (inputs.sum() + targets.sum() + smooth)
        return 1 - dice_score

class LossFactory(object):
    """A tool that construct a loss of model
    Parameters
    ----------
    loss_name : str
        name of loss. Defaults to 'binary_crossentropy'.
  
    """

    def __init__(self, loss_name='binary_crossentropy'):
        if loss_name in AVAILABLE_LOSS:
            self.loss_name = loss_name
        else:    
            raise NotImplementedError('{} has not been implemented, use loss in {}'.format(self.loss_name,AVAILABLE_LOSS))
        
    def get_loss_fn(self):        
        """get pytorch loss function
        Returns
        -------        
        torch.nn.losses            
            pytorch loss function        
        """        
        loss_dict = {'binary_crossentropy': BCELoss(), 'crossentropy': CrossEntropyLoss(), 'diceloss': DiceLoss()}
        return loss_dict.get(self.loss_name)


