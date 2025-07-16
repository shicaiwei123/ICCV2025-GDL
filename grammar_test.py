import argparse
import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import pdb

from dataset.CramedDataset import CramedDataset
from dataset.VGGSoundDataset import VGGSound
from dataset.dataset import AVDataset
# from models.basic_model import AVClassifier
from utils.utils import setup_seed, weight_init
import torch.nn.functional as F
import csv
import re


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='CREMAD', type=str,
                        help='VGGSound, KineticSound, CREMAD, AVE')
    parser.add_argument('--modulation', default='OGM_GE', type=str,

                        choices=['Normal', 'OGM', 'OGM_GE'])
    parser.add_argument('--fusion_method', default='concat', type=str,
                        choices=['sum', 'concat', 'gated', 'film'])
    parser.add_argument('--fps', default=1, type=int)
    parser.add_argument('--use_video_frames', default=3, type=int)
    parser.add_argument('--audio_path', default='/home/hudi/data/CREMA-D/AudioWAV', type=str)
    parser.add_argument('--visual_path', default='/home/hudi/data/CREMA-D/', type=str)

    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--epochs', default=100, type=int)

    parser.add_argument('--optimizer', default='sgd', type=str, choices=['sgd', 'adam'])
    parser.add_argument('--learning_rate', default=0.001, type=float, help='initial learning rate')
    parser.add_argument('--lr_decay_step', default=70, type=int, help='where learning rate decays')
    parser.add_argument('--lr_decay_ratio', default=0.1, type=float, help='decay coefficient')

    parser.add_argument('--modulation_starts', default=0, type=int, help='where modulation begins')
    parser.add_argument('--modulation_ends', default=50, type=int, help='where modulation ends')
    parser.add_argument('--alpha', required=True, type=float, help='alpha in OGM-GE')

    parser.add_argument('--ckpt_path', required=True, type=str, help='path to save trained models')
    parser.add_argument('--train', action='store_true', help='turn on train mode')

    parser.add_argument('--use_tensorboard', default=False, type=bool, help='whether to visualize')
    parser.add_argument('--tensorboard_path', type=str, help='path to save tensorboard logs')

    parser.add_argument('--random_seed', default=0, type=int)
    parser.add_argument('--gpu_ids', default='0, 1', type=str, help='GPU ids')

    return parser.parse_args()


def main():
    args = get_arguments()
    print(args)

    setup_seed(args.random_seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
    gpu_ids = list(range(torch.cuda.device_count()))

    device = torch.device('cuda:0')

    model = AVClassifier(args)
    model.apply(weight_init)
    model.to(device)

    model = torch.nn.DataParallel(model, device_ids=gpu_ids)

    if args.fusion_method == 'sum':
        out_v = torch.transpose(model.module.fusion_module.fc_y.weight, 0, 1)
        out_a = torch.transpose(model.module.fusion_module.fc_x.weight, 0, 1)
    else:
        weight_size = model.module.fusion_module.fc_out.weight.size(1)
        out_v = torch.transpose(model.module.fusion_module.fc_out.weight[:, weight_size // 2:], 0, 1)

        out_a = torch.transpose(model.module.fusion_module.fc_out.weight[:, :weight_size // 2], 0, 1)

    print(1)


def get_feature_diversity(a_feature):
    # a_feature=a_feature[0:2,:,:,:]
    # a_feature = a_feature.view(a_feature.shape[0], a_feature.shape[1], -1)  # B C HW
    # a_feature = a_feature.permute(0, 2, 1)  # B HW C
    # a_feature = a_feature - torch.mean(a_feature, dim=2, keepdim=True)
    # a_similarity = torch.bmm(a_feature, a_feature.permute(0, 2, 1))
    # a_std = torch.std(a_feature, dim=2)
    # a_std_matrix = torch.bmm(a_std.unsqueeze(dim=2), a_std.unsqueeze(dim=1))
    # a_similarity = a_similarity / a_std_matrix
    # # print(a_similarity)
    # # a_norm = torch.norm(a_similarity, dim=(1, 2)) / (a_similarity.shape[1] ** 2)
    # # print(a_norm.shape)
    # # a_norm = torch.mean(a_norm)
    # a_similarity = a_similarity.view(a_similarity.shape[0] * a_similarity.shape[1] * a_similarity.shape[2], 1)
    # return a_similarity
    G_s_average = 0
    for i in range(a_feature.shape[0]):
        feature_ones = a_feature[i, :]
        # print(feature_ones.shape)
        fm_s = feature_ones.view(feature_ones.shape[0], -1)

        fm_s_factors = torch.sqrt(torch.sum(fm_s * fm_s, 1))
        fm_s_trans = fm_s.t()
        fm_s_trans_factors = torch.sqrt(torch.sum(fm_s_trans * fm_s_trans, 0))
        # print(fm_s.shape,fm_s_factors.shape,fm_s_trans_factors.shape)
        fm_s_normal_factors = torch.mm(fm_s_factors.unsqueeze(1), fm_s_trans_factors.unsqueeze(0))
        G_s = torch.mm(fm_s, fm_s.t())
        G_s = (G_s / fm_s_normal_factors)

        # print(G_s.shape)

        G_s = G_s.view(G_s.shape[0] * G_s.shape[1], -1)
        G_s = G_s.squeeze(dim=1)
        G_s_average += G_s

    G_s_average = G_s_average / a_feature.shape[0]
    return G_s_average


if __name__ == "__main__":

    # a=np.array([[1,2,3,5],[6,7,8,9]])
    # a=a.transpose((1,0))
    # b=np.reshape(a,(2,4))
    #
    # a = torch.rand((64, 512, 7, 7)).float()
    # b = torch.rand((64, 512, 7, 7)).float()
    #
    # # a_simialr = get_feature_diversity(a)
    # # b_simialr = get_feature_diversity(b)
    # # print(a_simialr.mean())
    # # print(b_simialr.mean())
    #
    # # c=torch.randn_like(a)
    # # e=c[1,1,:,:]
    # #
    # # f=torch.randn((7,7))
    # # f_mean=torch.mean(f)
    # # f_std=torch.std(f)
    # #
    # # a_mean=torch.mean(c)
    # # a_std=torch.std(c)
    # # e_mean=torch.mean(e)
    # # e_std=torch.std(e)
    # # print(1)
    #
    # import re
    #
    # aa="1233455556"
    # b=aa[1:]
    #
    # # input_string = '"/abs"'
    # #
    # # # Use a regular expression to remove double quotes within single quotes
    # # output_string = re.sub(r'"(.*?)"', r'\1', input_string)
    # #
    # # print(output_string)
    #
    # input_string = '"/m/01g50p'
    #
    # input_string=input_string.strip('"')
    #
    # output_string = re.sub(r'^"\b', '', input_string)
    #
    # print(output_string)
    #
    # # print(aa==bb)
    # # ee=csv.reader(open("dataset/data/Aduioset/unbalanced_train_segments.csv"))
    # # for data in ee:
    # #     a=data
    # #     print(1)
    # # print(1)
    #
    # f = open("dataset/data/Aduioset/unbalanced_train_segments.csv")
    # f_class = open("dataset/data/Aduioset/class_labels_indices.csv")
    # data_list = f.readlines()[3:]
    # class_map = f_class.readlines()[1:]
    # video_name_to_class_dict = {}
    # class_label_dict = {}
    # for i in range(len(class_map)):
    #     data_i = class_map[i]
    #     data_i = data_i.split(',')
    #     video_name_to_class_dict.update({data_i[1]: data_i[2]})
    #     class_label_dict.update({data_i[2]: data_i[0]})
    #
    # # file_name_to_video_name_dict = {}
    # # for data in data_list:
    # #     data = data[0:-1]
    # #     data = data.split(',')
    # #     data[3] = data[3].strip()
    # #     output_string  = re.sub(r'"(.*?)"', r'\1', data[3])
    # #
    # #     output_string = output_string.strip('"')
    # #
    # #     file_name_to_video_name_dict.update({data[0]+'.wav': output_string})
    #
    #
    # dd=video_name_to_class_dict['/m/07hvw1']
    # print(2)


    # a=torch.acos(torch.tensor([0.001]))
    # print(1)

    a=torch.tensor([[2,2]]).float()
    b=a-2
    c=torch.cosine_similarity(a,a)
    d=a**2
    print(1)