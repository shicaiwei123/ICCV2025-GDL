import copy
import csv
import os
import pickle
import librosa
import numpy as np
from scipy import signal
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import pdb

class AVEDataset(Dataset):

    def __init__(self, args, mode='train'):
        self.args = args
        self.image = []
        self.audio = []
        self.label = []
        self.mode = mode
        classes = []

        self.data_root = './train_test_data/AVE_Dataset/'
        # class_dict = {'NEU':0, 'HAP':1, 'SAD':2, 'FEA':3, 'DIS':4, 'ANG':5}

        self.visual_feature_path = './train_test_data/AVE_Dataset'
        # self.audio_feature_path = './train_test_data/AVE_Dataset/Audio-1004-SE'
        self.audio_feature_path = './train_test_data/AVE_Dataset/Audios'


        self.train_txt = os.path.join(self.data_root,  'trainSet.txt')
        self.test_txt = os.path.join(self.data_root,  'testSet.txt')
        self.val_txt = os.path.join(self.data_root,  'valSet.txt')

        if mode == 'train':
            txt_file = self.train_txt
        elif mode == 'test':
            txt_file = self.test_txt
        else:
            txt_file = self.val_txt

        with open(self.test_txt, 'r') as f1:
            files = f1.readlines()
            for item in files:
                item = item.split('&')
                if item[0] not in classes:
                    classes.append(item[0])
        class_dict = {}
        for i, c in enumerate(classes):
            class_dict[c] = i

        with open(txt_file, 'r') as f2:
            files = f2.readlines()
            for item in files:
                item = item.split('&')
                audio_path = os.path.join(self.audio_feature_path, item[1] + '.wav')
                visual_path = os.path.join(self.visual_feature_path, 'Image-{:02d}-FPS-SE'.format(self.args.fps), item[1])

                # print(visual_path)
                # print(audio_path)

                if os.path.exists(audio_path) and os.path.exists(visual_path):
                    if os.stat(audio_path).st_size<200:
                        print(audio_path)
                        continue
                    if audio_path not in self.audio:
                        self.image.append(visual_path)
                        self.audio.append(audio_path)
                        self.label.append(class_dict[item[0]])
                else:
                    continue


    def __len__(self):
        return len(self.image)

    def __getitem__(self, idx):

        # # audio
        samples, rate = librosa.load(self.audio[idx], sr=22050)
        resamples = np.tile(samples, 3)[:22050*3]
        resamples[resamples > 1.] = 1.
        resamples[resamples < -1.] = -1.

        spectrogram = librosa.stft(resamples, n_fft=512, hop_length=256)
        spectrogram = np.log(np.abs(spectrogram) + 1e-7)
        spectrogram = np.resize(spectrogram, (224, 224))

        #mean = np.mean(spectrogram)
        #std = np.std(spectrogram)
        #spectrogram = np.divide(spectrogram - mean, std + 1e-9)


        # spectrogram = pickle.load(open(self.audio[idx], 'rb'))


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
        image_samples = os.listdir(self.image[idx])
        # select_index = np.random.choice(len(image_samples), size=self.args.num_frame, replace=False)
        # select_index.sort()
        images = torch.zeros((self.args.num_frame, 3, 224, 224))
        for i in range(self.args.num_frame):
            # for i, n in enumerate(select_index):
            img = Image.open(os.path.join(self.image[idx], image_samples[i])).convert('RGB')
            # print(os.path.join(self.image[idx], image_samples[i]))
            img = transform(img)
            images[i] = img

        images = torch.permute(images, (1,0,2,3))

        # label
        label = self.label[idx]

        # if spectrogram.shape !=(257, 1004):
        #     print(self.audio[idx])
        #     print(spectrogram.shape)
            # print("11111111111111")
        if images.shape !=torch.Size([3, self.args.num_frame, 224, 224]):
            print("22222222222")

        # print(spectrogram.shape,images.shape)

        return spectrogram, images, label