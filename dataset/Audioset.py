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
import re


def listdir_nohidden(path):
    return glob.glob(os.path.join(path, '*'))


def get_file_list(read_path):
    '''
    获取文件夹下图片的地址
    :param read_path:
    :return:
    '''
    path = read_path
    dirs = os.listdir(path)
    floder_len = len(dirs)
    file_name_list = []
    for i in range(floder_len):

        # 设置路径
        floder = dirs[i]
        floder_path = path + "/" + floder

        # 如果路径下是文件，那么就再次读取
        if os.path.isdir(floder_path):
            file_one = os.listdir(floder_path)
            file_len_one = len(file_one)
            for j in range(file_len_one):
                # 读取视频
                floder_path_one = floder_path + "/" + file_one[j]
                if os.path.isdir(floder_path_one):
                    file_two = os.listdir(floder_path_one)
                    file_len_two = len(file_two)
                    for k in range(file_len_two):
                        floder_path_two = floder_path_one + "/" + file_two[k]
                        if os.path.isdir(floder_path_two):
                            file_three = os.listdir(floder_path_two)
                            file_len_three = len(file_three)
                            for m in range(file_len_three):
                                floder_path_three = floder_path_two + "/" + file_three[m]
                                file_name_list.append(floder_path_three)
                        else:
                            file_name_list.append(floder_path_two)

                else:
                    file_name_list.append(floder_path_one)

        # 如果路径下，没有文件夹，直接是文件，就加入进来
        else:
            file_name_list.append(floder_path)

    return file_name_list


class AudioSet(nn.Module):
    def __init__(self, args, mode='train', data_path='./train_test_data/kinect_sound'):
        super().__init__()

        f_class = open("dataset/data/Aduioset/class_labels_indices.csv")

        class_map = f_class.readlines()[1:-1]
        self.video_name_to_class_dict = {}
        self.class_label_dict = {}
        for i in range(len(class_map)):
            data_i = class_map[i]
            data_i = data_i.split(',')
            self.video_name_to_class_dict.update({data_i[1]: data_i[2]})
            self.class_label_dict.update({data_i[2]: data_i[0]})
            print(1)

        self.file_name_to_video_name_dict_train = {}
        f = open("dataset/data/Aduioset/unbalanced_train_segments.csv")
        data_list = f.readlines()[3:-1]
        for data in data_list:
            data = data[0:-1]
            data = data.split(',')
            data[3] = data[3].strip()
            output_string = re.sub(r'"(.*?)"', r'\1', data[3])

            output_string = output_string.strip('"')

            self.file_name_to_video_name_dict_train.update({data[0] + '.wav': output_string})

        self.file_name_to_video_name_dict_test = {}
        f = open("dataset/data/Aduioset/eval_segments.csv")
        data_list = f.readlines()[3:-1]
        for data in data_list:
            data = data[0:-1]
            data = data.split(',')
            data[3] = data[3].strip()
            output_string = re.sub(r'"(.*?)"', r'\1', data[3])

            output_string = output_string.strip('"')

            self.file_name_to_video_name_dict_test.update({data[0] + '.wav': output_string})

        self.args = args

        # print(data_dict)

        self.mode = mode
        if self.mode == 'train':
            audio_data_path = os.path.join(data_path, 'train')
        elif self.mode == 'test':
            audio_data_path = os.path.join(data_path, 'test')

        self.file_list = get_file_list(audio_data_path)

        self.data_label = []

        # print(len(self.data_label))

    def __len__(self):
        # return 10000
        return len(self.file_list)

    def __getitem__(self, idx):

        # audio
        file_select = self.file_list[idx]
        sample, rate = librosa.load(file_select, sr=16000, mono=True)
        while len(sample) / rate < 10.:
            sample = np.tile(sample, 2)

        start_point = random.randint(a=0, b=rate * 5)
        new_sample = sample[start_point:start_point + rate * 5]
        new_sample[new_sample > 1.] = 1.
        new_sample[new_sample < -1.] = -1.

        spectrogram = librosa.stft(new_sample, n_fft=512, hop_length=256)
        spectrogram = np.log(np.abs(spectrogram) + 1e-7)
        print(spectrogram.shape)

        spectrogram = np.resize(spectrogram, (224, 224))
        spectrogram = np.reshape(spectrogram, (1, 224, 224))

        # label
        file_name = file_select.split('/')
        if self.mode == 'train':
            label = self.class_label_dict[
                self.video_name_to_class_dict[self.file_name_to_video_name_dict_train[file_name]]]
        else:
            label = self.class_label_dict[
                self.video_name_to_class_dict[self.file_name_to_video_name_dict_test[file_name]]]
        # print(label)

        return spectrogram, spectrogram, label
