import copy
import csv
import os
import pickle
import librosa
from scipy import signal
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import skimage
import random
import time
from PIL import Image, ImageFilter
import pdb
import torch.nn as nn
import glob
import numpy as np
import time


def listdir_nohidden(path):
    return glob.glob(os.path.join(path, '*'))


class KSDataset(nn.Module):
    def __init__(self, args, mode='train', data_path='./train_test_data/kinect_sound'):
        super().__init__()

        f = open('dataset/data/KineticSound/class.txt')
        data = f.readline()
        class_list = data.split(',')
        for i in range(len(class_list)):
            if " " in class_list[i]:
                class_name = class_list[i].split(" ")
                if class_name[0] == '':
                    class_name = class_name[1:len(class_name)]
                class_name = '_'.join(class_name)
                class_list[i] = class_name

        self.args = args

        # class_list=[class_list[0],class_list[1]]

        label = range(len(class_list))
        data_dict = zip(class_list, label)
        data_dict = dict(data_dict)

        # print(data_dict)

        self.mode = mode
        if self.mode == 'train':
            visual_data_path = os.path.join(data_path, 'visual', 'train_img/Image-01-FPS')
            audio_data_path = os.path.join(data_path, 'audio', 'train')
        elif self.mode == 'test':
            visual_data_path = os.path.join(data_path, 'visual', 'val_img/Image-01-FPS')
            audio_data_path = os.path.join(data_path, 'audio', 'test')

        self.data_label = []
        self.video_path_list = []
        self.audio_path_list = []

        remove_list = []  # 移除损坏视频

        # i=0
        for class_name in class_list:
            visual_class_path = os.path.join(visual_data_path, class_name)
            audio_class_path = os.path.join(audio_data_path, class_name)

            video_list = os.listdir(visual_class_path)
            video_list.sort()

            audio_list = os.listdir(audio_class_path)
            audio_list.sort()

            for video in video_list:
                # i+=1
                video_path = os.path.join(visual_class_path, video)

                if len(listdir_nohidden(video_path)) < 3:
                    # print(video_path)
                    remove_list.append(video)
                    continue

                self.video_path_list.append(video_path)
                self.data_label.append(data_dict[class_name])

            for audio in audio_list:
                if audio in remove_list:
                    print(audio)
                    continue
                audio_path = os.path.join(audio_class_path, audio)
                self.audio_path_list.append(audio_path)

        # print(len(self.data_label))

        # self.audio_path_list = self.audio_path_list[0:self.args.data_num]
        # self.video_path_list = self.video_path_list[0:self.args.data_num]
        # self.data_label = self.data_label[0:self.args.data_num]


        # audio_data = []
        # visual_data = []
        # label_data = []
        # count = torch.zeros(len(class_list))
        # for i in range(len(self.data_label)):
        #     label=self.data_label[i]
        #     if count[label] < self.args.data_num:
        #         audio_data.append(self.audio_path_list[i])
        #         visual_data.append(self.video_path_list[i])
        #         label_data.append(self.data_label[i])
        #         count[label] += 1

        # self.image = self.image[0:self.args.data_num]
        # self.label = self.label[0:self.args.data_num]
        # self.audio = self.audio[0:self.args.data_num]

        # self.video_path_list = visual_data
        # self.audio_path_list = audio_data
        # self.data_label = label_data

        # print("1",len(self.video_path_list))



    def __len__(self):
        # return 10000

        # if self.args.data_num < len(self.data_label):
        #
        #     return self.args.data_num
        # else:
        # print(len(self.data_label))
        return len(self.data_label)

    def __getitem__(self, idx):

        # audio
        sample, rate = librosa.load(self.audio_path_list[idx], sr=16000, mono=True)
        while len(sample) / rate < 10.:
            sample = np.tile(sample, 2)

        start_point = random.randint(a=0, b=rate * 5)
        new_sample = sample[start_point:start_point + rate * 5]
        new_sample[new_sample > 1.] = 1.
        new_sample[new_sample < -1.] = -1.

        spectrogram = librosa.stft(new_sample, n_fft=256, hop_length=128)
        spectrogram = np.log(np.abs(spectrogram) + 1e-7)
        spectrogram = np.transpose(spectrogram, (1, 0))
        # print(spectrogram.shape)

        # spectrogram=np.reshape(spectrogram,(spectrogram.shape[0]//2,spectrogram.shape[1]*2))

        spectrogram = np.transpose(spectrogram, (1, 0))
        # print(spectrogram.shape)

        # spectrogram = np.resize(spectrogram, (224, 224))
        # print(spectrogram.shape)

        if self.mode == 'train':
            transform = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        else:
            transform = transforms.Compose([
                transforms.Resize(size=(224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

        # Visual
        image_samples = listdir_nohidden(self.video_path_list[idx])
        # print(len(image_samples))
        select_index = np.random.choice(len(image_samples), size=self.args.use_video_frames, replace=False)
        select_index.sort()
        images = torch.zeros((self.args.use_video_frames, 3, 224, 224))
        for i in range(self.args.use_video_frames):
            try:
                img = Image.open(image_samples[i]).convert('RGB')
            except Exception as e:
                print(e)
                print(image_samples[i])
                continue

            bt = time.time()
            img = transform(img)
            et = time.time()
            # print(et-bt)
            images[i] = img

        images = torch.permute(images, (1, 0, 2, 3))

        # label
        label = self.data_label[idx]
        # print(label)

        return spectrogram, images, label
