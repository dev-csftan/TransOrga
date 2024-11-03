import os
import torch
from PIL import Image
from torchvision import transforms
from torch import nn
import numpy as np
import cv2


class Organoid(torch.utils.data.Dataset):
    def __init__(self,images,transform):
        self.images = images
        self.transform =transform

    def __getitem__(self, index):
        img =self.images[index]
        img =self.transform(img)
        #img =transforms.ToTensor()(img)
        return img
    def __len__(self):
        return len(self.images)
