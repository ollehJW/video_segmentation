import os
from PIL import Image
import cv2
from torch.utils.data import Dataset
from torchvision import transforms as T
import torch
import numpy as np

def prepare_dataset(image_dir = '', mask_dir = '', label_exist = True):
    """
    Prepare Dataset with Directory.
    Parameters
    ----------
    image_dir : str
        a directory of image files.
    mask_dir : str
        a directory of mask files.
    label_exist : bool
        Whether label exist.

    Returns
        Dictionary: Image list, Mask list
    """
    image_file_list = os.listdir(image_dir)
    image_file_list = [os.path.join(image_dir, fname) for fname in image_file_list]
    image_file_list = sorted(image_file_list)
    print("The number of image files: {}".format(len(image_file_list)))
    
    if label_exist:
        mask_file_list = os.listdir(mask_dir)
        mask_file_list = [os.path.join(mask_dir, fname) for fname in mask_file_list]
        mask_file_list = sorted(mask_file_list)
        print("The number of mask files: {}".format(len(mask_file_list)))
    else:
        mask_file_list = []

    
    return dict({'Image': image_file_list, 'Mask': mask_file_list})



class ConstructDataset(Dataset):
    """
    Construct Torch Dataset.
    Parameters
    ----------
    img_list : list
        a list of image 
    mask_list : str
        a directory of mask files.
    mean : float
        Mean of Nomalizaion parameters.
    std : float
        Std of Nomalizaion parameters.
    transform : Alburmentation
        Image Transformations.

    Returns
        image, mask
    """

    
    def __init__(self, img_list, mask_list, mean, std, transform=None):
        self.img_list = img_list
        self.mask_list = mask_list
        self.transform = transform
        self.mean = mean
        self.std = std
        
    def __len__(self):
        return len(self.img_list)
    
    def __getitem__(self, idx):
        img = cv2.imread(self.img_list[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = np.array(Image.open(self.mask_list[idx]))
        mask[mask == 255] = 0
        
        if self.transform is not None:
            aug = self.transform(image=img, mask=mask)
            img = Image.fromarray(aug['image'])
            mask = aug['mask']
        
        if self.transform is None:
            img = Image.fromarray(img)
        
        t = T.Compose([T.ToTensor(), T.Normalize(self.mean, self.std)])
        img = t(img)
        mask = torch.from_numpy(mask).long()
        return img, mask

    
    
class ConstructInferenceDataset(Dataset):

    """
    Construct Torch Dataset.
    Parameters
    ----------
    img_list : list
        a list of image 
    mask_list : str
        a directory of mask files.
    mean : float
        Mean of Nomalizaion parameters.
    std : float
        Std of Nomalizaion parameters.
    transform : Alburmentation
        Image Transformations.

    Returns
        image, mask
    """

    def __init__(self, img_list, mask_list, transform=None):
        self.img_list = img_list
        self.mask_list = mask_list
        self.transform = transform
      
    def __len__(self):
        return len(self.img_list)
    
    def __getitem__(self, idx):
        img = cv2.imread(self.img_list[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = np.array(Image.open(self.mask_list[idx]))
        mask[mask == 255] = 0
        
        if self.transform is not None:
            aug = self.transform(image=img, mask=mask)
            img = Image.fromarray(aug['image'])
            mask = aug['mask']
        
        if self.transform is None:
            img = Image.fromarray(img)
        
        mask = torch.from_numpy(mask).long()
        
        return img, mask