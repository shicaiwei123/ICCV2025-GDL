import argparse
import os
import pstats

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
import pdb
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.lr_scheduler import ReduceLROnPlateau

from dataset.CramedDataset import CramedDataset
from dataset.KSDataset import KSDataset
from dataset.VGGSoundDataset import VGGSound
from dataset.AVEDataset import AVEDataset
from models.basic_model import AVClassifier, AVClassifier_SWIN
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

    parser.add_argument('--audio_path', default='./data/CREMA-D/AudioWAV', type=str)
    parser.add_argument('--visual_path', default='./data/CREMA-D', type=str)

    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--epochs', default=100, type=int)

    parser.add_argument('--optimizer', default='sgd', type=str)
    parser.add_argument('--learning_rate', default=0.002, type=float, help='initial learning rate')
    parser.add_argument('--lr_decay_step', default='[30,70]', type=str, help='where learning rate decays')
    parser.add_argument('--lr_decay_ratio', default=0.1, type=float, help='decay coefficient')

    parser.add_argument('--modulation_starts', default=0, type=int, help='where modulation begins')
    parser.add_argument('--modulation_ends', default=50, type=int, help='where modulation ends')
    parser.add_argument('--alpha', required=True, type=float, help='alpha in OGM-GE')

    parser.add_argument('--ckpt_path', required=True, type=str, help='path to save trained models')
    parser.add_argument('--train', action='store_true', help='turn on train mode')

    parser.add_argument('--use_tensorboard', default=False, type=bool, help='whether to visualize')
    parser.add_argument('--tensorboard_path', type=str, help='path to save tensorboard logs')

    parser.add_argument('--random_seed', default=0, type=int)
    parser.add_argument('--gpu_ids', default='1', type=str, help='GPU ids')
    parser.add_argument('--pe', type=int, default=0)
    parser.add_argument('--max', type=int, default=1e20)
    parser.add_argument('--modality', type=str, default='full')
    parser.add_argument('--beta', type=float, default=0)
    parser.add_argument('--pretrain', type=bool, default=False)
    parser.add_argument('--backbone', type=str, default='resnet')
    parser.add_argument('--total_epoch', default=10, type=int)
    parser.add_argument('--warmup', type=bool, default=False)
    parser.add_argument('--gamma', type=float, default=1.0)
    parser.add_argument('--drop', default=0, type=int)

    return parser.parse_args()


def get_feature_diversity(a_feature):
    a_feature = a_feature.view(a_feature.shape[0], a_feature.shape[1], -1)  # B C HW
    a_feature = a_feature.permute(0, 2, 1)  # B HW C
    a_feature = a_feature - torch.mean(a_feature, dim=2, keepdim=True)
    a_similarity = torch.bmm(a_feature, a_feature.permute(0, 2, 1))
    a_std = torch.std(a_feature, dim=2)
    a_std_matrix = torch.bmm(a_std.unsqueeze(dim=2), a_std.unsqueeze(dim=1))
    a_similarity = a_similarity / a_std_matrix
    # print(a_similarity)
    a_norm = torch.norm(a_similarity, dim=(1, 2)) / (a_similarity.shape[1] ** 2)
    # print(a_norm.shape)
    a_norm = torch.mean(a_norm)
    return a_norm


def regurize(mul, std):
    variance_dul = std ** 2
    variance_dul = variance_dul.view(variance_dul.shape[0], -1)
    mul = mul.view(mul.shape[0], -1)
    loss_kl = ((variance_dul + mul ** 2 - torch.log((variance_dul + 1e-8)) - 1) * 0.5)

    loss_kl = torch.sum(loss_kl, dim=1)

    loss_kl = torch.mean(loss_kl)

    return loss_kl


def get_feature_diff(x1, x2):
    # print(x1.shape,x2.shape)
    x1 = F.adaptive_avg_pool2d(x1, (7, 7))
    x2 = F.adaptive_avg_pool2d(x2, (7, 7))
    # x1 = torch.mean(x1, dim=(2, 3))
    # x2 = torch.mean(x2, dim=(2, 3))

    x1 = x1.permute(0, 2, 3, 1).contiguous()
    x2 = x2.permute(0, 2, 3, 1).contiguous()

    rgb = x1.view(-1, x1.shape[3])
    depth = x2.view(-1, x2.shape[3])

    diff = F.mse_loss(rgb, depth)
    # diff = torch.cosine_similarity(rgb, depth)
    # diff = torch.mean(diff)
    # print(simi.shape)
    return diff


def train_epoch(args, epoch, model, device, dataloader, optimizer, scheduler, scheduler_warmup, writer=None):
    criterion = nn.CrossEntropyLoss()
    softmax = nn.Softmax(dim=1)
    relu = nn.ReLU(inplace=True)
    tanh = nn.Tanh()

    if scheduler_warmup is not None:
        scheduler_warmup.step(epoch=epoch + 1)
    elif scheduler is not None:
        scheduler.step()

    if epoch < 20:
        print(epoch, optimizer.param_groups[0]['lr'])

    model.train()
    print("Start training ... ")

    _loss = 0
    _loss_a = 0
    _loss_v = 0
    _a_diveristy = 0
    _v_diveristy = 0
    _a_re = 0
    _v_re = 0
    similar_average = 0

    log_list=[]

    for step, (spec, image, label) in enumerate(
            tqdm(dataloader, desc="Epoch {}/{}".format(epoch, args.epochs))):

        # pdb.set_trace()
        spec = spec.to(device)
        image = image.to(device)
        label = label.to(device)

        optimizer.zero_grad()

        # TODO: make it simpler and easier to extend
        a, v, out, a_feature, v_feature, a_mul, a_std, v_mul, v_std, out_a, out_v = model(spec.unsqueeze(1).float(),
                                                                                          image.float())

        # print(a_feature.shape,v_feature.shape)

        # similar = get_feature_diff(a_feature, v_feature)
        similar_average += 0
        # print(similar.mean())

        # loss_v = criterion(out_v, label)
        # loss_a = criterion(out_a, label)
        loss_f = criterion(out, label)

        # loss_cls = loss_f + (loss_a + loss_v) * args.gamma

        loss_cls = loss_f 
        loss_a=loss_f
        loss_v=loss_f

        a_diveristy = get_feature_diversity(a_feature)
        v_diveristy = get_feature_diversity(v_feature)

        # if epoch<2:
        #     a_std = torch.clamp(a_std, min=0, max=2)
        #     v_std = torch.clamp(v_std, min=0, max=2)

        # print(a_mul)
        if not isinstance(a_mul, int):
            regurize_a = regurize(a_mul, a_std)
            regurize_a = regurize_a.cuda()
        else:
            regurize_a = torch.zeros(1).float().cuda()

        if not isinstance(v_mul, int):
            regurize_v = regurize(v_mul, v_std)
            regurize_v = regurize_v.cuda()
            # print(regurize_a)
        else:
            regurize_v = torch.zeros(1).float().cuda()

        # if epoch < 2:
        #     regurize_loss = torch.zeros(1).float().cuda()
        # else:
        #     regurize_loss = (regurize_a + regurize_v) * args.beta

        regurize_loss = (regurize_a + regurize_v)
        # if regurize_loss>10:
        #     regurize_loss=regurize_loss/(regurize_loss/10.0)

        loss = loss_cls + regurize_loss * args.beta
        # print(loss)
        if step % 100 == 0:
            # print(a_std.mean().item(),v_std.mean().item())
            print("regurize_Loss:", regurize_loss.item(), "unimodal_loss:", (loss_a + loss_v).item(), "cls_loss:",
                  loss_cls.item())

        # calculate_a = torch.mean(out_a, 0).sum().cpu().detach().numpy()
        # calculate_b = torch.mean(out_v, 0).sum().cpu().detach().numpy()
        # print("calculate:", calculate_a, calculate_b)
        #
        # fc_weight = model.module.fusion_module.fc_out.weight
        # fc_weight = fc_weight.T
        #
        # fc_weight_mean = fc_weight[:, 3]
        #
        # visual = torch.mean(torch.abs(fc_weight_mean[0:512]))
        # audio = torch.mean(torch.abs(fc_weight_mean[512:1024]))
        #
        # print("weight:", torch.sum(audio).cpu().detach().numpy(), torch.sum(visual).cpu().detach().numpy())

        # with open("weight_of_a_v.csv", 'a', newline='') as f:
        #     writer = csv.writer(f)
        #     row = [calculate_a, calculate_b]
        #     writer.writerow(row)

        loss.backward()

        nn.utils.clip_grad_norm_(model.parameters(), max_norm=40, norm_type=2)

        # audio_grad_sum = 0
        # index=0
        # for p in model.module.audio_net.parameters():
        #     index+=1
        #     # print(p.grad)
        #     audio_grad_sum += torch.abs(p.grad).mean().item()
        #
        # visual_grad_sum = 0
        # index=0
        # for p in model.module.visual_net.parameters():
        #     index+=1
        #     visual_grad_sum += torch.abs(p.grad).mean().item()
        #
        # print("grad:",audio_grad_sum, visual_grad_sum)
        #
        # for p in model.module.visual_net.parameters():
        #     p.grad = p.grad * 10

        if args.modulation == 'Normal':
            # no modulation, regular optimization
            pass
        else:
            pass

            # audio_grad_sum = 0
            # count = 0
            # for p in model.module.audio_net.parameters():
            #     audio_grad_sum += torch.abs(p.grad).mean().item()
            #     p.grad = p.grad
            #     count += 1
            #
            # visual_grad_sum = 0
            # for p in model.module.visual_net.parameters():
            #     count += 1
            #     visual_grad_sum += torch.abs(p.grad).mean().item()
            #
            # for p in model.module.audio_net.parameters():
            #     p.grad = p.grad * visual_grad_sum / audio_grad_sum + torch.zeros_like(p.grad).normal_(0,
            #                                                                                           p.grad.std().item() + 1e-8)
            # for p in model.module.visual_net.parameters():
            #     p.grad = p.grad * audio_grad_sum / visual_grad_sum + torch.zeros_like(p.grad).normal_(0,
            #                                                                                           p.grad.std().item() + 1e-8)

            # # Modulation starts here !
            # score_v = sum([softmax(out_v)[i][label[i]] for i in range(out_v.size(0))])
            # score_a = sum([softmax(out_a)[i][label[i]] for i in range(out_a.size(0))])
            #
            # ratio_v = score_v / score_a
            # ratio_a = 1 / ratio_v
            #
            # """
            # Below is the Eq.(10) in our CVPR paper:
            #         1 - tanh(alpha * rho_t_u), if rho_t_u > 1
            # k_t_u =
            #         1,                         else
            # coeff_u is k_t_u, where t means iteration steps and u is modality indicator, either a or v.
            # """
            #
            # if ratio_v > 1:
            #     coeff_v = 1 - tanh(args.alpha * relu(ratio_v))
            #     coeff_a = 1
            # else:
            #     coeff_a = 1 - tanh(args.alpha * relu(ratio_a))
            #     coeff_v = 1
            #
            # if args.use_tensorboard:
            #     iteration = epoch * len(dataloader) + step
            #     writer.add_scalar('data/ratio v', ratio_v, iteration)
            #     writer.add_scalar('data/coefficient v', coeff_v, iteration)
            #     writer.add_scalar('data/coefficient a', coeff_a, iteration)
            #
            # if args.modulation_starts <= epoch <= args.modulation_ends:  # bug fixed
            #     for name, parms in model.named_parameters():
            #         layer = str(name).split('.')[1]
            #
            #         if 'audio' in layer and len(parms.grad.size()) == 4:
            #             if args.modulation == 'OGM_GE':  # bug fixed
            #                 parms.grad = parms.grad * coeff_a + \
            #                              torch.zeros_like(parms.grad).normal_(0, parms.grad.std().item() + 1e-8)
            #             elif args.modulation == 'OGM':
            #                 parms.grad *= coeff_a
            #
            #         if 'visual' in layer and len(parms.grad.size()) == 4:
            #             if args.modulation == 'OGM_GE':  # bug fixed
            #                 parms.grad = parms.grad * coeff_v + \
            #                              torch.zeros_like(parms.grad).normal_(0, parms.grad.std().item() + 1e-8)
            #             elif args.modulation == 'OGM':
            #                 parms.grad *= coeff_v
            # else:
            #     pass

        optimizer.step()

        _loss += loss.item()
        _loss_a += loss_a.item()
        _loss_v += loss_v.item()
        _a_diveristy += a_diveristy.item()
        _v_diveristy += v_diveristy.item()
        _a_re += regurize_a.item()
        _v_re += regurize_v.item()

        # if step % 100 == 0:
        #     print(step, loss)

    similar_average = similar_average / (step + 1)
    print("mse_diff:", similar_average)
    # print(regurize_v,regurize_a)
    # file_name = 'audio_visual_similar_in_numtimodal' + '.csv'
    # with open(file_name, 'a', newline='') as f:
    #     writer = csv.writer(f)
    #     row = [similar_average.cpu().detach().numpy()]
    #     writer.writerow(row)

    return _loss / len(dataloader), _loss_a / len(dataloader), _loss_v / len(dataloader), _a_diveristy / len(
        dataloader), _v_diveristy / len(dataloader), _a_re / len(dataloader), _v_re / len(dataloader),


class GradualWarmupScheduler(_LRScheduler):
    """ Gradually warm-up(increasing) learning rate in optimizer.
    Proposed in 'Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour'.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        multiplier: target learning rate = base lr * multiplier if multiplier > 1.0. if multiplier = 1.0, lr starts from 0 and ends up with the base_lr.
        total_epoch: target learning rate is reached at total_epoch, gradually
        after_scheduler: after target_epoch, use this scheduler(eg. ReduceLROnPlateau)
    """

    def __init__(self, args, optimizer, multiplier, after_scheduler=None):
        self.multiplier = multiplier
        if self.multiplier < 1.:
            raise ValueError('multiplier should be greater thant or equal to 1.')
        self.total_epoch = args.total_epoch
        self.after_scheduler = after_scheduler
        self.finished = False
        super(GradualWarmupScheduler, self).__init__(optimizer)

    def get_lr(self):
        if self.last_epoch > self.total_epoch:
            if self.after_scheduler:
                if not self.finished:
                    self.after_scheduler.base_lrs = [base_lr * self.multiplier for base_lr in self.base_lrs]
                    self.finished = True
                # return self.after_scheduler.get_last_lr()
                return [group['lr'] for group in self.optimizer.param_groups]
            return [base_lr * self.multiplier for base_lr in self.base_lrs]

        if self.multiplier == 1.0:
            return [base_lr * (float(self.last_epoch) / self.total_epoch) for base_lr in self.base_lrs]
        else:
            return [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in
                    self.base_lrs]

    def step_ReduceLROnPlateau(self, metrics, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch if epoch != 0 else 1  # ReduceLROnPlateau is called at the end of epoch, whereas others are called at beginning
        if self.last_epoch <= self.total_epoch:
            warmup_lr = [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in
                         self.base_lrs]
            for param_group, lr in zip(self.optimizer.param_groups, warmup_lr):
                param_group['lr'] = lr
        else:
            if epoch is None:
                self.after_scheduler.step(metrics, None)
            else:
                self.after_scheduler.step(metrics, epoch - self.total_epoch)

    def step(self, epoch=None, metrics=None):
        if type(self.after_scheduler) != ReduceLROnPlateau:
            if self.finished and self.after_scheduler:
                if epoch is None:
                    self.after_scheduler.step(None)
                else:
                    self.after_scheduler.step(epoch - self.total_epoch)
                self._last_lr = [group['lr'] for group in self.optimizer.param_groups]
            else:
                return super(GradualWarmupScheduler, self).step(epoch)
        else:
            self.step_ReduceLROnPlateau(metrics, epoch)


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

            a, v, out, a_feature, v_feature, _, _, _, _,out_a, out_v = model(spec.unsqueeze(1).float(), image.float())
            
            out_v=out
            out_a=out

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


def main():
    args = get_arguments()
    args.p = [0, 0]
    print(args)

    setup_seed(args.random_seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
    gpu_ids = list(range(torch.cuda.device_count()))

    device = torch.device('cuda:0')

    if args.backbone == 'resnet':
        model = AVClassifier(args)
        model.apply(weight_init)

    elif args.backbone == 'swin':
        model = AVClassifier_SWIN(args)
    else:
        raise EOFError

    model.to(device)

    model = torch.nn.DataParallel(model, device_ids=gpu_ids)

    model.cuda()

    if args.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, eval(args.lr_decay_step), args.lr_decay_ratio)

    elif args.optimizer == 'AdaGrad':
        optimizer = optim.Adagrad(model.parameters(), lr=args.learning_rate)
        scheduler = None
    elif args.optimizer == 'Adam':
        optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.999))
        scheduler = None
    else:
        raise ValueError

    if args.warmup:
        scheduler_warmup = GradualWarmupScheduler(args, optimizer, multiplier=1,
                                                  after_scheduler=scheduler)
    else:
        scheduler_warmup = None

    if args.dataset == 'VGGSound':
        train_dataset = VGGSound(args, mode='train')
        test_dataset = VGGSound(args, mode='test')
    elif args.dataset == 'KineticSound':
        train_dataset = KSDataset(args, mode='train')
        test_dataset = KSDataset(args, mode='test')
    elif args.dataset == 'kinect400':
        train_dataset = Kinect400(args, mode='train')
        test_dataset = Kinect400(args, mode='test')
    elif args.dataset == 'CREMAD':
        train_dataset = CramedDataset(args, mode='train')
        test_dataset = CramedDataset(args, mode='test')
    elif args.dataset == 'AVE':
        train_dataset = AVEDataset(args, mode='train')
        test_dataset = AVEDataset(args, mode='test')
    else:
        raise NotImplementedError('Incorrect dataset name {}! '
                                  'Only support VGGSound, KineticSound and CREMA-D for now!'.format(args.dataset))

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size,
                                  shuffle=True, num_workers=32, pin_memory=True, drop_last=True)

    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size,
                                 shuffle=False, num_workers=32, pin_memory=True, drop_last=True)


    log_path = os.path.join(args.ckpt_path, args.dataset)

    if not os.path.exists(log_path):
        os.makedirs(log_path)

    save_path=os.path.join(log_path,args.modality + '.csv')
    with open(save_path, 'a+', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=",")
        writer.writerow([1000, 1000, 1000])
    if args.train:

        best_acc = 0.0

        for epoch in range(args.epochs):

            print('Epoch: {}: '.format(epoch))
            args.epoch_now = epoch

            if args.use_tensorboard:

                writer_path = os.path.join(args.tensorboard_path, args.dataset)
                if not os.path.exists(writer_path):
                    os.mkdir(writer_path)
                log_name = '{}_{}'.format(args.fusion_method, args.modulation)
                writer = SummaryWriter(os.path.join(writer_path, log_name))

                batch_loss, batch_loss_a, batch_loss_v, a_diveristy, v_diveristy, a_re, v_re = train_epoch(args,
                                                                                                           epoch,
                                                                                                           model,
                                                                                                           device,
                                                                                                           train_dataloader,
                                                                                                           optimizer,
                                                                                                           scheduler,
                                                                                                           scheduler_warmup)
                acc, acc_a, acc_v = valid(args, model, device, test_dataloader)

                writer.add_scalars('Loss', {'Total Loss': batch_loss,
                                            'Audio Loss': batch_loss_a,
                                            'Visual Loss': batch_loss_v}, epoch)

                writer.add_scalars('Evaluation', {'Total Accuracy': acc,
                                                  'Audio Accuracy': acc_a,
                                                  'Visual Accuracy': acc_v}, epoch)

            else:
                batch_loss, batch_loss_a, batch_loss_v, a_diveristy, v_diveristy, a_re, v_re = train_epoch(args, epoch,
                                                                                                           model,
                                                                                                           device,
                                                                                                           train_dataloader,
                                                                                                           optimizer,
                                                                                                           scheduler,
                                                                                                           scheduler_warmup)
                acc, acc_a, acc_v = valid(args, model, device, test_dataloader)

                with open(save_path, 'a+', newline='') as csvfile:
                    writer = csv.writer(csvfile, delimiter=",")
                    writer.writerow([acc, acc_a, acc_v])

            if acc > best_acc and epoch:
                best_acc = float(acc)

                if not os.path.exists(args.ckpt_path):
                    os.makedirs(args.ckpt_path)

                model_name = 'best_model_of_dataset_{}_{}_alpha_{}_pe_{}_beta{}_' \
                             'optimizer_{}_modulate_starts_{}_ends_{}_' \
                             'epoch_{}_acc_{}.pth'.format(args.dataset,
                                                          args.modulation,
                                                          args.alpha,
                                                          args.pe,
                                                          args.beta,
                                                          args.optimizer,
                                                          args.modulation_starts,
                                                          args.modulation_ends,
                                                          epoch, acc)

                if scheduler is None:
                    saved_dict = {'saved_epoch': epoch,
                                  'modulation': args.modulation,
                                  'alpha': args.alpha,
                                  'fusion': args.fusion_method,
                                  'acc': acc,
                                  'model': model.state_dict(),
                                  'optimizer': optimizer.state_dict(),
                                  }
                else:
                    saved_dict = {'saved_epoch': epoch,
                                  'modulation': args.modulation,
                                  'alpha': args.alpha,
                                  'fusion': args.fusion_method,
                                  'acc': acc,
                                  'model': model.state_dict(),
                                  'optimizer': optimizer.state_dict(),
                                  'scheduler': scheduler.state_dict()}

                save_dir = os.path.join(args.ckpt_path, model_name)

                # torch.save(saved_dict, save_dir)
                print('The best model has been saved at {}.'.format(save_dir))
                print("Loss: {:.3f}, Acc: {:.3f}".format(batch_loss, acc))
                print("Audio Acc: {:.3f}， Visual Acc: {:.3f} ".format(acc_a, acc_v))
                print("Audio similar: {:.3f}， Visual similar: {:.3f} ".format(a_diveristy, v_diveristy))
                print("Audio regurize: {:.3f}， Visual regurize: {:.3f} ".format(a_re, v_re))
            else:
                print("Loss: {:.3f}, Acc: {:.3f}, Best Acc: {:.3f}".format(batch_loss, acc, best_acc))
                print("Audio Acc: {:.3f}， Visual Acc: {:.3f} ".format(acc_a, acc_v))
                print("Audio similar: {:.3f}， Visual similar: {:.3f} ".format(a_diveristy, v_diveristy))
                print("Audio regurize: {:.3f}， Visual regurize: {:.3f} ".format(a_re, v_re))

    else:
        # first load trained model
        loaded_dict = torch.load(args.ckpt_path)
        # epoch = loaded_dict['saved_epoch']
        modulation = loaded_dict['modulation']
        # alpha = loaded_dict['alpha']
        fusion = loaded_dict['fusion']
        state_dict = loaded_dict['model']
        # optimizer_dict = loaded_dict['optimizer']
        # scheduler = loaded_dict['scheduler']

        assert modulation == args.modulation, 'inconsistency between modulation method of loaded model and args !'
        assert fusion == args.fusion_method, 'inconsistency between fusion method of loaded model and args !'
        # print(state_dict)
        model.load_state_dict(state_dict)
        # model.train()
        # model.eval()
        print('Trained model loaded!')

        acc, acc_a, acc_v = valid(args, model, device, test_dataloader)
        print('Accuracy: {}, accuracy_a: {}, accuracy_v: {}'.format(acc, acc_a, acc_v))


if __name__ == "__main__":
    main()
