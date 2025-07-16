import argparse
import os
import pstats

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F

from dataset.CramedDataset import CramedDataset,CramedDataset_swin
from dataset.KSDataset import KSDataset
from dataset.VGGSoundDataset import VGGSound
from dataset.AVEDataset import AVEDataset
from models.basic_model import AVClassifier_DGL
from utils.utils import setup_seed, weight_init
from dataset.Kinect400 import Kinect400
import csv
import numpy as np
from tqdm import tqdm


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
    parser.add_argument('--num_frame', default=1, type=int, help='use how many frames for train')

    parser.add_argument('--audio_path', default='./train_test_data/CREMA-D/AudioWAV', type=str)
    parser.add_argument('--visual_path', default='./train_test_data/CREMA-D', type=str)

    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--epochs', default=100, type=int)

    parser.add_argument('--optimizer', default='sgd', type=str)
    parser.add_argument('--learning_rate', default=0.001, type=float, help='initial learning rate')
    parser.add_argument('--lr_decay_step', default='[70]', type=str, help='where learning rate decays')
    parser.add_argument('--lr_decay_ratio', default=0.1, type=float, help='decay coefficient')

    parser.add_argument('--modulation_starts', default=0, type=int, help='where modulation begins')
    parser.add_argument('--modulation_ends', default=50, type=int, help='where modulation ends')
    parser.add_argument('--alpha', default=4.0, type=float, help='alpha in DGL')

    parser.add_argument('--ckpt_path', required=True, type=str, help='path to save trained models')
    parser.add_argument('--train', action='store_true', help='turn on train mode')

    parser.add_argument('--use_tensorboard', default=False, type=bool, help='whether to visualize')
    parser.add_argument('--tensorboard_path', type=str, help='path to save tensorboard logs')

    parser.add_argument('--random_seed', default=0, type=int)
    parser.add_argument('--gpu_ids', default='1', type=str, help='GPU ids')
    parser.add_argument('--modality', type=str, default='full')
    parser.add_argument('--backbone', type=str, default='resnet')
    parser.add_argument('--total_epoch', default=10, type=int)
    parser.add_argument('--drop', default=0, type=int)

    return parser.parse_args()

def valid(args, model, device, dataloader):
    softmax = nn.Softmax(dim=1)

    if args.dataset == 'VGGSound':
        n_classes = 309
    elif args.dataset == 'KineticSound':
        n_classes = 34
    elif args.dataset == 'kinect400':
        n_classes = 400
    elif args.dataset == 'CREMAD':
        n_classes = 6
    elif args.dataset == 'AVE':
        n_classes = 28
    else:
        raise NotImplementedError('Incorrect dataset name {}'.format(args.dataset))

    model.module.args.drop = 0
    with torch.no_grad():
        model.eval()
        # TODO: more flexible
        print(model.module.args.drop)
        num = [0.0 for _ in range(n_classes)]
        acc = [0.0 for _ in range(n_classes)]
        acc_a = [0.0 for _ in range(n_classes)]
        acc_v = [0.0 for _ in range(n_classes)]

        for step, (spec, image, label) in enumerate(dataloader):

            spec = spec.to(device)
            image = image.to(device)
            label = label.to(device)

            out, out_a, out_v = model(spec.unsqueeze(1).float(), image.float())

            prediction = softmax(out)
            pred_v = softmax(out_v)
            pred_a = softmax(out_a)

            for i in range(image.shape[0]):

                ma = np.argmax(prediction[i].cpu().data.numpy())
                v = np.argmax(pred_v[i].cpu().data.numpy())
                a = np.argmax(pred_a[i].cpu().data.numpy())
                num[label[i]] += 1.0

                # pdb.set_trace()
                if np.asarray(label[i].cpu()) == ma:
                    acc[label[i]] += 1.0
                if np.asarray(label[i].cpu()) == v:
                    acc_v[label[i]] += 1.0
                if np.asarray(label[i].cpu()) == a:
                    acc_a[label[i]] += 1.0

    model.module.args.drop = 1
    return sum(acc) / sum(num), sum(acc_a) / sum(num), sum(acc_v) / sum(num)

def modal_main():
    args = get_arguments()
    print(args)

    setup_seed(args.random_seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
    gpu_ids = list(range(torch.cuda.device_count()))

    device = torch.device('cuda:0')

    model = AVClassifier_DGL(args)


    audio_model.apply(weight_init)
    audio_model.to(device)

    visual_model.apply(weight_init)
    visual_model.to(device)

    audio_model = torch.nn.DataParallel(audio_model, device_ids=gpu_ids)

    visual_model = torch.nn.DataParallel(visual_model, device_ids=gpu_ids)

    model_path = "./results/best_model_of_dataset_CREMAD_Normal_alpha_0.8_optimizer_sgd_modulate_starts_0_ends_50_epoch_75_acc_0.6290322580645161.pth"

    model.load_state_dict(torch.load(model_path)['model'], strict=False)

    model.cuda()

    if args.dataset == 'VGGSound':
        train_dataset = VGGSound(args, mode='train')
        test_dataset = VGGSound(args, mode='test')
    elif args.dataset == 'KineticSound':
        train_dataset = KSDataset(args, mode='train')
        test_dataset = KSDataset(args, mode='test')
    elif args.dataset == 'CREMAD':
        train_dataset = CramedDataset(args, mode='train')
        test_dataset = CramedDataset(args, mode='test')
    elif args.dataset == 'AVE':
        train_dataset = AVEDataset(args, mode='train')
        test_dataset = AVEDataset(args, mode='test')
    else:
        raise NotImplementedError('Incorrect dataset name {}! '
                                  'Only support VGGSound, KineticSound and CREMA-D for now!'.format(args.dataset))

    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size,
                                 shuffle=False, num_workers=32, pin_memory=True)

    acc, acc_a, acc_v = valid(args,model, device, test_dataloader)

    print('Accuracy: {}, accuracy_a: {}, accuracy_v: {}'.format(acc, acc_a, acc_v))


if __name__ == '__main__':
    # two_modal_main()
    modal_main()
