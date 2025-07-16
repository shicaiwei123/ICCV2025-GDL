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


class Kinect400(nn.Module):
    def __init__(self, args, mode='train', data_path='./train_test_data/kinect400'):
        super().__init__()

        # f = open('dataset/data/KineticSound/class.txt')
        # data = f.readline()
        # class_list = data.split(',')
        # for i in range(len(class_list)):
        #     if " " in class_list[i]:
        #         class_name = class_list[i].split(" ")
        #         if class_name[0] == '':
        #             class_name = class_name[1:len(class_name)]
        #         class_name = '_'.join(class_name)
        #         class_list[i] = class_name
        class_name_list = os.listdir(os.path.join(data_path, 'audio/train'))
        class_name_list.sort()
        class_name_list=class_name_list[0:300]
        self.class_name_to_label = {}
        for i in range(len(class_name_list)):
            self.class_name_to_label.update({class_name_list[i]: i})
        self.args = args

        # print(data_dict)

        self.mode = mode
        if self.mode == 'train':
            visual_data_path = os.path.join(data_path, 'visual', 'train_img/Image-01-FPS')
            audio_data_path = os.path.join(data_path, 'audio', 'train')
        elif self.mode == 'test':
            visual_data_path = os.path.join(data_path, 'visual', 'val_img/Image-01-FPS/')
            audio_data_path = os.path.join(data_path, 'audio', 'test')

        self.data_label = []
        self.video_path_list = []
        self.audio_path_list = []

        remove_list = []  # 移除损坏视频

        # i=0
        for class_name in class_name_list:
            visual_class_path = os.path.join(visual_data_path, class_name)
            audio_class_path = os.path.join(audio_data_path, class_name)

            video_list = os.listdir(visual_class_path)
            video_list.sort()

            audio_list = os.listdir(audio_class_path)
            audio_list.sort()

            for audio in audio_list:
                audio_path = os.path.join(audio_class_path, audio)
                if os.stat(audio_path).st_size < 80 * 1024:
                    # print("audio_less", audio_path)
                    continue
                video_path = os.path.join(visual_class_path, audio.split('.')[0])
                if len(listdir_nohidden(video_path)) < 3:
                    # print("visual less", video_path)
                    continue

                try:
                    self.data_label.append(self.class_name_to_label[class_name])
                    self.audio_path_list.append(audio_path)
                    self.video_path_list.append(video_path)
                except Exception as e:
                    print("not in class_dict")

            # for audio in audio_list:
            #
            #     audio_path = os.path.join(audio_class_path, audio)
            #     if os.stat(audio_path).st_size < 100 * 1024:
            #         remove_list.append(audio)
            #         continue
            #     self.audio_path_list.append(audio_path)
            #
            # for video in video_list:
            #     # i+=1
            #     video_path = os.path.join(visual_class_path, video)
            #     if len(listdir_nohidden(video_path)) < 3:
            #         print("video_less3", video_path)
            #     if video in remove_list:
            #         continue
            #
            #     self.video_path_list.append(video_path)
            #     self.data_label.append(self.class_name_to_label[class_name])

        print(len(self.video_path_list), len(self.audio_path_list), len(self.data_label))

    def __len__(self):
        # return 10000
        return len(self.data_label)

    def __getitem__(self, idx):

        # audio
        sample, rate = librosa.load(self.audio_path_list[idx], sr=16000, mono=True)
        while len(sample) / rate < 10.:
            sample = np.tile(sample, 2)

        start_point = random.randint(a=0, b=rate * 2)
        new_sample = sample[start_point:start_point + rate * 8]
        new_sample[new_sample > 1.] = 1.
        new_sample[new_sample < -1.] = -1.

        spectrogram = librosa.stft(new_sample, n_fft=256, hop_length=128)
        spectrogram = np.log(np.abs(spectrogram) + 1e-7)

        # spectrogram = np.resize(spectrogram, (224, 224))
        # spectrogram = (spectrogram - (-1.6)) / 2.0
        # print(np.mean(spectrogram), np.std(spectrogram))

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
