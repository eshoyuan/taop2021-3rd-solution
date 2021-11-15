import os
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import numpy as np
import glob
from PIL import Image
import random
from pydicom import dcmread


# data augmentation
from datasets.augmentations import get_hard_augmentations, get_hard_augmentations_v2, get_medium_augmentations, \
    get_light_augmentations
import albumentations as A
import cv2
from albumentations import pytorch
from albumentations import (
    HorizontalFlip, VerticalFlip, IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90,
    Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,
    IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, IAAPiecewiseAffine, RandomResizedCrop,
    IAASharpen, IAAEmboss, RandomBrightnessContrast, Flip, OneOf, Compose, Normalize, Cutout, CoarseDropout,
    ShiftScaleRotate, CenterCrop, Resize
)


class Ti_jpg(Dataset):
    def __init__(self, image_list, label, size, phase, augmentation=0):

        super(Ti_jpg, self).__init__()
        self.Image_files = image_list
        self.Label_files = label
        self.phase = phase
        self.augmentation = augmentation
        self.size = size

    def __len__(self):
        return len(self.Image_files)

    # Kaggle2019 APTOS 2019 Blindness Detection 7th 方法, test使用augmentation = 1, test使用augmentation = 2
    @classmethod
    def preprocess(self, img, augmentation, image_size=(448, 448)):
        if augmentation == 1:
            transform = A.Compose([A.LongestMaxSize(image_size[0], interpolation=cv2.INTER_CUBIC),
                                   A.PadIfNeeded(image_size[0], image_size[1], border_mode=cv2.BORDER_CONSTANT,
                                                 value=0),
                                   get_medium_augmentations(image_size),
                                   A.Normalize(),
                                   A.pytorch.ToTensorV2()
                                   ])
        elif augmentation == 2:
            transform = A.Compose([
                A.LongestMaxSize(image_size[0], interpolation=cv2.INTER_CUBIC),
                A.PadIfNeeded(image_size[0], image_size[1],
                              border_mode=cv2.BORDER_CONSTANT, value=0),
                HorizontalFlip(p=0.5),
                VerticalFlip(p=0.5),

                A.Normalize(),
                A.pytorch.ToTensorV2()])
        elif augmentation == 3:
            transform = A.Compose([A.LongestMaxSize(image_size[0], interpolation=cv2.INTER_CUBIC),
                                   A.PadIfNeeded(image_size[0], image_size[1], border_mode=cv2.BORDER_CONSTANT,
                                                 value=0),
                                   get_light_augmentations(image_size),
                                   A.Normalize(),
                                   A.pytorch.ToTensorV2()
                                   ])
        img = transform(image=img)
        img = img["image"]
        return img

    def __getitem__(self, i):
        img = cv2.imread(self.Image_files[i])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = self.preprocess(img, augmentation=self.augmentation,
                              image_size=(self.size, self.size))

        label = int(self.Label_files[i]) - 1
        label = torch.tensor(label)
        return img.float(), label.long()


class Ti_jpg_onehot(Dataset):
    def __init__(self, image_list, label, phase, augmentation=0):

        super(Ti_jpg_onehot, self).__init__()
        self.Image_files = image_list
        self.Label_files = label
        self.phase = phase
        self.augmentation = augmentation

    def __len__(self):
        return len(self.Image_files)

    @ classmethod
    def preprocess(self, img, augmentation, image_size=(448, 448)):
        if augmentation == 1:
            transform = A.Compose([A.LongestMaxSize(image_size[0], interpolation=cv2.INTER_CUBIC),
                                   A.PadIfNeeded(image_size[0], image_size[1], border_mode=cv2.BORDER_CONSTANT,
                                                 value=0),
                                   get_medium_augmentations(image_size),
                                   # get_hard_augmentations_v2(), get_light_augmentations()
                                  A.Normalize(),
                                  A.pytorch.ToTensorV2()
                                   ])
        elif augmentation == 2:
            transform = A.Compose([
                A.LongestMaxSize(image_size[0], interpolation=cv2.INTER_CUBIC),
                A.PadIfNeeded(image_size[0], image_size[1],
                              border_mode=cv2.BORDER_CONSTANT, value=0),
                HorizontalFlip(p=0.5),
                VerticalFlip(p=0.5),

                A.Normalize(),
                A.pytorch.ToTensorV2()])
        img = transform(image=img)  # 转换图片为符合网络输入的大小的tensor
        img = img["image"]
        return img

    def __getitem__(self, i):
        img = cv2.imread(self.Image_files[i])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = self.preprocess(img, augmentation=self.augmentation)

        grade = []
        for j in range(1, 6):
            if j == int(self.Label_files[i]):
                grade.append(1)
            else:
                grade.append(0)
        grade = np.array(grade)

        return img, torch.from_numpy(grade).float()


class Ti_jpg_predict(Dataset):
    def __init__(self, image_list, size, augmentation=0):
        self.Image_files = image_list
        self.augmentation = augmentation
        self.size = size

    def __len__(self):
        return len(self.Image_files)

    @ classmethod
    def preprocess(self, img, augmentation, image_size=(448, 448)):
        if augmentation == 1:
            transform = A.Compose([A.LongestMaxSize(image_size[0], interpolation=cv2.INTER_CUBIC),
                                   A.PadIfNeeded(image_size[0], image_size[1], border_mode=cv2.BORDER_CONSTANT,
                                                 value=0),
                                   get_medium_augmentations(image_size),
                                   # get_hard_augmentations_v2(), get_light_augmentations()
                                  A.Normalize(),
                                  A.pytorch.ToTensorV2()
                                   ])
        elif augmentation == 2:
            transform = A.Compose([
                A.LongestMaxSize(image_size[0], interpolation=cv2.INTER_CUBIC),
                A.PadIfNeeded(image_size[0], image_size[1],
                              border_mode=cv2.BORDER_CONSTANT, value=0),
                HorizontalFlip(p=0.5),
                VerticalFlip(p=0.5),

                A.Normalize(),
                A.pytorch.ToTensorV2()])
        img = transform(image=img)
        img = img["image"]
        return img

    def __getitem__(self, i):
        img = cv2.imread(self.Image_files[i])
        # print(self.Image_files[i])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = self.preprocess(img, augmentation=self.augmentation,
                              image_size=(self.size, self.size))

        label = 0  # 为了和train的时候保持一样的格式, label并无实际作用
        label = torch.tensor(label)
        return img.float(), label.long()
