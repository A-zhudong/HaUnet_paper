import logging
from os import listdir
from os.path import splitext
from pathlib import Path
import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
import cv2 as cv


def collate_pool(datasetlist):
    '''
    get the images and labels correctly and send the length out
    return the stack of the labels and the batch of images
    '''
    labels = []; images = []; coords = []
    for i, (image, label) in enumerate(datasetlist):
        label = label.long()
        shape = image.shape
        # coords.append(torch.round(label[:,:2]/(torch.round(torch.max(label[:,2])*2/150)*150/2)*511))
        coords.append(label[:2,:].T)
        labels.append(label[2,:])
        # onehot_labels = np.zeros(labels.shape[0],4).scatter_(1, labels.type(torch.LongTensor), 1)
        images.append(image)
    labels = torch.cat(labels, dim=0)
    images = torch.stack(images, dim=0).unsqueeze(dim=1)
    # print(images.shape)
    # print(coords[0])
    return coords, labels, images/torch.max(images)
    

class ImageDataset(Dataset):
    # for npy format and different labels
    def __init__(self, images_dir: str, labels_dir: str):
        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir)
        self.ids = np.loadtxt(os.path.join(self.labels_dir, 'index.txt'),dtype=np.str)
    
    def __len__(self):
        return(len(self.ids))
    
    def __getitem__(self, index):
        idx = int(float(self.ids[index]))
        imagePath = os.path.join(self.images_dir,'{}.png'.format(idx))
        img = cv.imread(imagePath, cv.IMREAD_GRAYSCALE)
        # label = np.loadtxt(os.path.join(self.labels_dir, '{}.txt'.format(idx)))
        # change to npy
        label = np.load(os.path.join(self.labels_dir, '{}.npy'.format(idx)))
        # print(int(float(self.ids[index])),np.max(label[:2,:]),np.min(label[:2,:]),np.round(np.max(label[:2,:])*2/150))
        # label[:2,:] = np.round(label[:2,:].copy()/(np.round(np.max(label[:2,:])*2/150)*150/2)*511)
        image = torch.as_tensor(img.copy()).float().contiguous()
        # label = torch.as_tensor(label.copy()).int().t().contiguous()
        label = torch.as_tensor(label.copy()).t().int().contiguous()
        return image, label
      
class HardAttenImageDataset(Dataset):
    def __init__(self, images_dir: str, labels_dir: str):
        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir)
        self.ids = np.loadtxt(os.path.join(self.labels_dir, 'index.txt'),dtype=np.str)
    
    def __len__(self):
        return(len(self.ids))
    
    def __getitem__(self, index):
        idx = int(float(self.ids[index]))
        imagePath = os.path.join(self.images_dir,'{}.npy'.format(idx))
        img = np.load(imagePath)
        label = np.loadtxt(os.path.join(self.labels_dir, '{}.txt'.format(idx)))
        # print(int(float(self.ids[index])),np.max(label[:2,:]),np.min(label[:2,:]),np.round(np.max(label[:2,:])*2/150))
        # label[:2,:] = np.round(label[:2,:].copy()/(np.round(np.max(label[:2,:])*2/150)*150/2)*511)
        image = torch.as_tensor(img.T.copy()).contiguous() #convert to (H, W) mode
        label = torch.as_tensor(label.copy()).int().contiguous()
        return image, label

class BasicDataset(Dataset):
    def __init__(self, images_dir: str, masks_dir: str, scale: float = 1.0, mask_suffix: str = ''):
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'
        self.scale = scale
        self.mask_suffix = mask_suffix

        self.ids = [splitext(file)[0] for file in listdir(images_dir) if not file.startswith('.')]
        if not self.ids:
            raise RuntimeError(f'No input file found in {images_dir}, make sure you put your images there')
        logging.info(f'Creating dataset with {len(self.ids)} examples')

    def __len__(self):
        return len(self.ids)

    @staticmethod
    def preprocess(pil_img, scale, is_mask):
        w, h = pil_img.size
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small, resized images would have no pixel'
        pil_img = pil_img.resize((newW, newH), resample=Image.NEAREST if is_mask else Image.BICUBIC)
        img_ndarray = np.asarray(pil_img)

        if not is_mask:
            if img_ndarray.ndim == 2:
                img_ndarray = img_ndarray[np.newaxis, ...]
            else:
                img_ndarray = img_ndarray.transpose((2, 0, 1))

            img_ndarray = img_ndarray / 255

        return img_ndarray

    @staticmethod
    def load(filename):
        ext = splitext(filename)[1]
        if ext in ['.npz', '.npy']:
            return Image.fromarray(np.load(filename))
        elif ext in ['.pt', '.pth']:
            return Image.fromarray(torch.load(filename).numpy())
        else:
            return Image.open(filename)

    def __getitem__(self, idx):
        name = self.ids[idx]
        mask_file = list(self.masks_dir.glob(name + self.mask_suffix + '.*'))
        img_file = list(self.images_dir.glob(name + '.*'))

        assert len(img_file) == 1, f'Either no image or multiple images found for the ID {name}: {img_file}'
        assert len(mask_file) == 1, f'Either no mask or multiple masks found for the ID {name}: {mask_file}'
        mask = self.load(mask_file[0])
        img = self.load(img_file[0])

        assert img.size == mask.size, \
            f'Image and mask {name} should be the same size, but are {img.size} and {mask.size}'

        img = self.preprocess(img, self.scale, is_mask=False)
        mask = self.preprocess(mask, self.scale, is_mask=True)

        return {
            'image': torch.as_tensor(img.copy()).float().contiguous(),
            'mask': torch.as_tensor(mask.copy()).long().contiguous()
        }


class CarvanaDataset(BasicDataset):
    def __init__(self, images_dir, masks_dir, scale=1):
        super().__init__(images_dir, masks_dir, scale, mask_suffix='_mask')
