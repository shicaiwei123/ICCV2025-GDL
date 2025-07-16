
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as tt
from lib.processing_utils import get_file_list, get_mean_std
import cv2
from PIL import Image
import os
import numpy as np


class ImgBinaryDataset(Dataset):

    def __init__(self, living_dir, spoofing_dir, balance=True, data_transform=None):

        self.living_path_list = get_file_list(living_dir)
        self.spoofing_path_list = get_file_list(spoofing_dir)

        # 间隔取样,控制数量
        if balance:
            self.spoofing_path_list = sorted(self.spoofing_path_list)
            balance_factor = int(np.floor(len(self.spoofing_path_list) / len(self.living_path_list)))
            if balance_factor==0:
                balance_factor=1
            self.spoofing_path_list = self.spoofing_path_list[0:len(self.spoofing_path_list):balance_factor]
        self.img_path_list = self.spoofing_path_list + self.living_path_list
        self.data_transform = data_transform

    def __getitem__(self, idx):
        img_path = self.img_path_list[idx]
        face_img = cv2.imread(img_path)

        if self.data_transform is not None:
            face_img_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
            face_img_pil = Image.fromarray(face_img_rgb)
            face_img_trans = self.data_transform(face_img_pil)
            face_img = face_img_trans

        # 确定label
        img_path_split = img_path.split('/')
        if img_path_split[-3] == 'spoofing' or img_path_split[-2]:
            label = 0
        else:
            label = 1


        return face_img, label

    def __len__(self):
        return len(self.img_path_list)




class ImgPixelDataset(Dataset):

    def __init__(self, living_dir, spoofing_dir, balance=True, data_transform=None, pixel_size=32):

        self.living_path_list = get_file_list(living_dir)
        self.spoofing_path_list = get_file_list(spoofing_dir)

        # 间隔取样,控制数量
        if balance:
            self.spoofing_path_list = sorted(self.spoofing_path_list)
            balance_factor = int(np.floor(len(self.spoofing_path_list) / len(self.living_path_list)))
            if balance_factor == 0:
                balance_factor = 1
            self.spoofing_path_list = self.spoofing_path_list[0:len(self.spoofing_path_list):balance_factor]
        self.img_path_list = self.spoofing_path_list + self.living_path_list
        self.data_transform = data_transform
        self.pixel_size = pixel_size

    def __getitem__(self, idx):
        img_path = self.img_path_list[idx]
        face_img = cv2.imread(img_path)
        face_img_pil=Image.open(img_path).convert('RGB')

        if self.data_transform is not None:
            # face_img_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
            # face_img_pil = Image.fromarray(face_img_rgb)
            face_img_trans = self.data_transform(face_img_pil)
            face_img = face_img_trans

        # 确定label
        img_path_split = img_path.split('/')
        if img_path_split[-3] == 'spoofing' or img_path_split[-2]=='spoofing':
            label = np.float32(np.zeros((self.pixel_size, self.pixel_size)))
        else:
            label = np.float32(np.ones((self.pixel_size, self.pixel_size)))

        return face_img, label

    def __len__(self):
        return len(self.img_path_list)



