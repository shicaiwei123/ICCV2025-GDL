'''模型训练相关的函数'''

import numpy as np
import torch
import torch.optim as optim
import torch.utils.data
from tqdm import tqdm
import time
import csv
import os
from torchtoolbox.tools import mixup_criterion, mixup_data
import time

import os
import torch.nn as nn
import torch.nn.functional as F

from lib.model_develop_utils import GradualWarmupScheduler, calc_accuracy
from loss.mmd_loss import MMD_loss
import datetime
from datasets.surf_txt import SURF, SURF_generate
from src.surf_baseline_multi_dataloader import surf_multi_transforms_train, surf_multi_transforms_test
from lib.processing_utils import get_dataself_hist


def calc_accuracy_multi_advisor(model, loader, args, verbose=False, hter=False):
    """
    :param model: model network
    :param loader: torch.utils.data.DataLoader
    :param verbose: show progress bar, bool
    :return accuracy, float
    """
    mode_saved = model.training
    model.train(False)
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        model.cuda()
    outputs_full = []
    labels_full = []
    for sample_batch in tqdm(iter(loader), desc="Full forward pass", total=len(loader), disable=not verbose):
        img_rgb, ir_target, depth_target, binary_target = sample_batch['image_rgb'], sample_batch['image_ir'], \
            sample_batch['image_depth'], sample_batch['binary_label']

        if torch.cuda.is_available():
            img_rgb, ir_target, depth_target, binary_target = img_rgb.cuda(), ir_target.cuda(), depth_target.cuda(), binary_target.cuda()

        with torch.no_grad():
            if args.method == 'deeppix':
                ir_out, depth_out, outputs_batch = model(img_rgb)
            elif args.method == 'pyramid':
                if args.origin_deeppix:
                    x, x, x, x, outputs_batch = model(img_rgb)
                else:
                    x, x, x, x, outputs_batch = model(img_rgb)
            else:
                print("test error")
        outputs_full.append(outputs_batch)
        labels_full.append(binary_target)
    model.train(mode_saved)
    outputs_full = torch.cat(outputs_full, dim=0)
    labels_full = torch.cat(labels_full, dim=0)
    _, labels_predicted = torch.max(outputs_full.data, dim=1)
    accuracy = torch.sum(labels_full == labels_predicted).item() / float(len(labels_full))
    accuracy = float("%.6f" % accuracy)

    if hter:
        predict_arr = np.array(labels_predicted.cpu())
        label_arr = np.array(labels_full.cpu())

        living_wrong = 0  # living -- spoofing
        living_right = 0
        spoofing_wrong = 0  # spoofing ---living
        spoofing_right = 0

        for i in range(len(predict_arr)):
            if predict_arr[i] == label_arr[i]:
                if label_arr[i] == 1:
                    living_right += 1
                else:
                    spoofing_right += 1
            else:
                # 错误
                if label_arr[i] == 1:
                    living_wrong += 1
                else:
                    spoofing_wrong += 1

        FRR = living_wrong / (living_wrong + living_right)
        APCER = living_wrong / (spoofing_right + living_wrong)
        NPCER = spoofing_wrong / (spoofing_wrong + living_right)
        FAR = spoofing_wrong / (spoofing_wrong + spoofing_right)
        HTER = (FAR + FRR) / 2

        FAR = float("%.6f" % FAR)
        FRR = float("%.6f" % FRR)
        HTER = float("%.6f" % HTER)
        accuracy = float("%.6f" % accuracy)

        return [accuracy, FAR, FRR, HTER, APCER, NPCER]
    else:
        return [accuracy]


def calc_accuracy_multi(model, loader, verbose=False, hter=False):
    """
    :param model: model network
    :param loader: torch.utils.data.DataLoader
    :param verbose: show progress bar, bool
    :return accuracy, float
    """
    mode_saved = model.training
    model.train(False)
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        model.cuda()
    outputs_full = []
    labels_full = []

    mul_full = []
    std_full = []

    for batch_sample in tqdm(iter(loader), desc="Full forward pass", total=len(loader), disable=not verbose):

        img_rgb, img_ir, img_depth, target = batch_sample['image_x'], batch_sample['image_ir'], \
            batch_sample['image_depth'], batch_sample[
            'binary_label']

        if torch.cuda.is_available():
            img_rgb = img_rgb.cuda()
            img_ir = img_ir.cuda()
            img_depth = img_depth.cuda()
            target = target.cuda()

        with torch.no_grad():
            outputs_batchs = model(img_rgb, img_ir, img_depth)
            if isinstance(outputs_batchs, tuple):
                outputs_batch = outputs_batchs[0]
            # print(outputs_batch)
        outputs_full.append(outputs_batch)
        labels_full.append(target)

        # (mul, std) = outputs_batchs[-1]
        # # mul_full.append(mul)
        # if torch.sum(std) != 0:
        #     std_concat = True
        #     std_full.append(std)
        #     # mul_full.append(mul)
        # else:
        #     std_concat = False

    model.train(mode_saved)
    outputs_full = torch.cat(outputs_full, dim=0)
    labels_full = torch.cat(labels_full, dim=0)

    # if std_concat:
    #     std_full = torch.cat(std_full, dim=0)
    #     # mul_full = torch.cat(mul_full, dim=0)

    _, labels_predicted = torch.max(outputs_full.data, dim=1)
    accuracy = torch.sum(labels_full == labels_predicted).item() / float(len(labels_full))
    # print((labels_full - labels_predicted))
    accuracy = float("%.6f" % accuracy)

    if hter:
        predict_arr = np.array(labels_predicted.cpu())
        label_arr = np.array(labels_full.cpu())

        living_wrong = 0  # living -- spoofing
        living_right = 0
        spoofing_wrong = 0  # spoofing ---living
        spoofing_right = 0

        for i in range(len(predict_arr)):
            if predict_arr[i] == label_arr[i]:
                if label_arr[i] == 1:
                    living_right += 1
                else:
                    spoofing_right += 1
            else:
                # 错误
                if label_arr[i] == 1:
                    living_wrong += 1
                else:
                    spoofing_wrong += 1

        try:

            FRR = living_wrong / (living_wrong + living_right)
            APCER = living_wrong / (spoofing_right + living_wrong)
            NPCER = spoofing_wrong / (spoofing_wrong + living_right)
            FAR = spoofing_wrong / (spoofing_wrong + spoofing_right)
            HTER = (FAR + FRR) / 2

            ACER = (APCER + NPCER) / 2

            FAR = float("%.6f" % FAR)
            FRR = float("%.6f" % FRR)
            HTER = float("%.6f" % HTER)
            accuracy = float("%.6f" % accuracy)
        except Exception as e:
            print(living_right, living_wrong, spoofing_right, spoofing_wrong), (
                outputs_full, labels_full, mul_full, std_full)
            return [accuracy, 0, 0, 0, 0, 0], (outputs_full, labels_full, mul_full, std_full)

        print(living_right, living_wrong, spoofing_right, spoofing_wrong)
        return [accuracy, FAR, FRR, HTER, APCER, NPCER, ACER], (outputs_full, labels_full, mul_full, std_full)
    else:
        return [accuracy], (outputs_full, labels_full, mul_full, std_full)


def calc_accuracy_ensemble(model, loader, verbose=False, hter=False):
    """
    :param model: model network
    :param loader: torch.utils.data.DataLoader
    :param verbose: show progress bar, bool
    :return accuracy, float
    """
    mode_saved = model.training
    model.train(False)
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        model.cuda()
    outputs_full = []
    labels_full = []

    for batch_sample in tqdm(iter(loader), desc="Full forward pass", total=len(loader), disable=not verbose):

        img_rgb, img_ir, img_depth, target = batch_sample['image_x'], batch_sample['image_ir'], \
            batch_sample['image_depth'], batch_sample[
            'binary_label']

        if torch.cuda.is_available():
            img_rgb = img_rgb.cuda()
            img_ir = img_ir.cuda()
            img_depth = img_depth.cuda()
            target = target.cuda()

        with torch.no_grad():

            outputs_batch = model(img_rgb)

            # 如果有多个返回值只取第一个
            if isinstance(outputs_batch, tuple):
                outputs_batch = outputs_batch[0]
            # print(outputs_batch)
        outputs_full.append(outputs_batch)
        labels_full.append(target)

    model.train(mode_saved)
    outputs_full = torch.cat(outputs_full, dim=0)
    labels_full = torch.cat(labels_full, dim=0)
    _, labels_predicted = torch.max(outputs_full.data, dim=1)
    accuracy = torch.sum(labels_full == labels_predicted).item() / float(len(labels_full))
    # print((labels_full - labels_predicted))
    accuracy = float("%.6f" % accuracy)

    if hter:
        predict_arr = np.array(labels_predicted.cpu())
        label_arr = np.array(labels_full.cpu())

        living_wrong = 0  # living -- spoofing
        living_right = 0
        spoofing_wrong = 0  # spoofing ---living
        spoofing_right = 0

        for i in range(len(predict_arr)):
            if predict_arr[i] == label_arr[i]:
                if label_arr[i] == 1:
                    living_right += 1
                else:
                    spoofing_right += 1
            else:
                # 错误
                if label_arr[i] == 1:
                    living_wrong += 1
                else:
                    spoofing_wrong += 1

        try:

            FRR = living_wrong / (living_wrong + living_right)
            APCER = living_wrong / (spoofing_right + living_wrong)
            NPCER = spoofing_wrong / (spoofing_wrong + living_right)
            ACER = (APCER + NPCER) / 2
            FAR = spoofing_wrong / (spoofing_wrong + spoofing_right)
            HTER = (FAR + FRR) / 2

            FAR = float("%.6f" % FAR)
            FRR = float("%.6f" % FRR)
            HTER = float("%.6f" % HTER)
            APCER = float("%.6f" % APCER)
            NPCER = float("%.6f" % NPCER)
            ACER = float("%.6f" % ACER)
            accuracy = float("%.6f" % accuracy)
        except Exception as e:
            print(e)
            print(living_right, living_wrong, spoofing_right, spoofing_wrong)
            return [accuracy, 1, 1, 1, 1, 1, 1]

        print(living_right, living_wrong, spoofing_right, spoofing_wrong)
        return [accuracy, FAR, FRR, HTER, APCER, NPCER, ACER]
    else:
        return [accuracy]


def calc_accuracy_kd(model, loader, args, verbose=False, hter=False):
    """
    :param model: model network
    :param loader: torch.utils.data.DataLoader
    :param verbose: show progress bar, bool
    :return accuracy, float
    """
    mode_saved = model.training
    model.train(False)
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        model.cuda()
    outputs_full = []
    labels_full = []

    if args.student_data == 'multi_rgb' or args.student_data == 'multi_depth' or args.student_data == 'multi_ir':
        for batch_sample in tqdm(iter(loader), desc="Full forward pass", total=len(loader), disable=not verbose):

            img_rgb, img_ir, img_depth, target = batch_sample['image_x'], batch_sample['image_ir'], \
                batch_sample['image_depth'], batch_sample[
                'binary_label']

            if torch.cuda.is_available():
                img_rgb = img_rgb.cuda()
                img_ir = img_ir.cuda()
                img_depth = img_depth.cuda()
                target = target.cuda()

            if args.student_data == 'multi_rgb':
                test_data = img_rgb
            elif args.student_data == 'multi_depth':
                test_data = img_depth
            elif args.student_data == 'multi_ir':
                test_data = img_ir
            else:
                test_data = img_rgb
                print('test_error')
            with torch.no_grad():
                outputs_batch = model(test_data)

                # 如果有多个返回值只取第一个
                if isinstance(outputs_batch, tuple):
                    outputs_batch = outputs_batch[0]
            outputs_full.append(outputs_batch)
            labels_full.append(target)
    else:
        for (batch_sample, single_sample, label) in tqdm(iter(loader), desc="Full forward pass", total=len(loader),
                                                         disable=not verbose):

            if torch.cuda.is_available():
                single_sample = single_sample.cuda()
                label = label.cuda()

            with torch.no_grad():
                outputs_batch = model(single_sample)

                # 如果有多个返回值只取第一个
                if isinstance(outputs_batch, tuple):
                    outputs_batch = outputs_batch[0]

                # print(outputs_batch)
            outputs_full.append(outputs_batch)
            labels_full.append(label)

    model.train(mode_saved)
    outputs_full = torch.cat(outputs_full, dim=0)
    labels_full = torch.cat(labels_full, dim=0)
    _, labels_predicted = torch.max(outputs_full.data, dim=1)
    accuracy = torch.sum(labels_full == labels_predicted).item() / float(len(labels_full))
    # print((labels_full - labels_predicted))
    accuracy = float("%.6f" % accuracy)

    if hter:
        predict_arr = np.array(labels_predicted.cpu())
        label_arr = np.array(labels_full.cpu())

        living_wrong = 0  # living -- spoofing
        living_right = 0
        spoofing_wrong = 0  # spoofing ---living
        spoofing_right = 0

        for i in range(len(predict_arr)):
            if predict_arr[i] == label_arr[i]:
                if label_arr[i] == 1:
                    living_right += 1
                else:
                    spoofing_right += 1
            else:
                # 错误
                if label_arr[i] == 1:
                    living_wrong += 1
                else:
                    spoofing_wrong += 1

        try:

            FRR = living_wrong / (living_wrong + living_right)
            APCER = living_wrong / (spoofing_right + living_wrong)
            NPCER = spoofing_wrong / (spoofing_wrong + living_right)
            FAR = spoofing_wrong / (spoofing_wrong + spoofing_right)
            HTER = (FAR + FRR) / 2

            FAR = float("%.6f" % FAR)
            FRR = float("%.6f" % FRR)
            HTER = float("%.6f" % HTER)
            accuracy = float("%.6f" % accuracy)
        except Exception as e:
            print(living_right, living_wrong, spoofing_right, spoofing_wrong)
            return [accuracy, 0, 0, 0, 0, 0]

        print(living_right, living_wrong, spoofing_right, spoofing_wrong)
        return [accuracy, FAR, FRR, HTER, APCER, NPCER]
    else:
        return [accuracy]


def calc_accuracy_kd_admd(model, loader, args, verbose=False, hter=False):
    """
    :param model: model network
    :param loader: torch.utils.data.DataLoader
    :param verbose: show progress bar, bool
    :return accuracy, float
    """
    mode_saved = model.training
    model.train(False)
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        model.cuda()
    outputs_full = []
    labels_full = []

    for (img_rgb, img_ir, img_depth, target) in tqdm(iter(loader), desc="Full forward pass", total=len(loader),
                                                     disable=not verbose):

        if torch.cuda.is_available():
            img_rgb = img_rgb.cuda()
            img_ir = img_ir.cuda()
            img_depth = img_depth.cuda()
            target = target.cuda()

        with torch.no_grad():
            outputs_batch = model(img_rgb)

            # 如果有多个返回值只取第一个
            if isinstance(outputs_batch, tuple):
                outputs_batch = outputs_batch[0]

            # print(outputs_batch)
        outputs_full.append(outputs_batch)
        labels_full.append(target)

    model.train(mode_saved)
    outputs_full = torch.cat(outputs_full, dim=0)
    labels_full = torch.cat(labels_full, dim=0)
    _, labels_predicted = torch.max(outputs_full.data, dim=1)
    accuracy = torch.sum(labels_full == labels_predicted).item() / float(len(labels_full))
    # print((labels_full - labels_predicted))
    accuracy = float("%.6f" % accuracy)

    if hter:
        predict_arr = np.array(labels_predicted.cpu())
        label_arr = np.array(labels_full.cpu())

        living_wrong = 0  # living -- spoofing
        living_right = 0
        spoofing_wrong = 0  # spoofing ---living
        spoofing_right = 0

        for i in range(len(predict_arr)):
            if predict_arr[i] == label_arr[i]:
                if label_arr[i] == 1:
                    living_right += 1
                else:
                    spoofing_right += 1
            else:
                # 错误
                if label_arr[i] == 1:
                    living_wrong += 1
                else:
                    spoofing_wrong += 1

        try:

            FRR = living_wrong / (living_wrong + living_right)
            APCER = living_wrong / (spoofing_right + living_wrong)
            NPCER = spoofing_wrong / (spoofing_wrong + living_right)
            FAR = spoofing_wrong / (spoofing_wrong + spoofing_right)
            HTER = (FAR + FRR) / 2

            FAR = float("%.6f" % FAR)
            FRR = float("%.6f" % FRR)
            HTER = float("%.6f" % HTER)
            accuracy = float("%.6f" % accuracy)
        except Exception as e:
            print(living_right, living_wrong, spoofing_right, spoofing_wrong)
            return [accuracy, 0, 0, 0, 0, 0]

        print(living_right, living_wrong, spoofing_right, spoofing_wrong)
        return [accuracy, FAR, FRR, HTER, APCER, NPCER]
    else:
        return [accuracy]


def calc_accuracy_kd_patch(model, loader, args, verbose=False, hter=False):
    """
    :param model: model network
    :param loader: torch.utils.data.DataLoader
    :param verbose: show progress bar, bool
    :return accuracy, float
    """
    mode_saved = model.training
    model.train(False)
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        model.cuda()
    outputs_full = []
    labels_full = []

    if args.student_data == 'multi_rgb':
        for batch_sample in tqdm(iter(loader), desc="Full forward pass", total=len(loader), disable=not verbose):

            img_rgb, img_ir, img_depth, target = batch_sample['image_x'], batch_sample['image_ir'], \
                batch_sample['image_depth'], batch_sample[
                'binary_label']

            if torch.cuda.is_available():
                img_rgb = img_rgb.cuda()
                img_ir = img_ir.cuda()
                img_depth = img_depth.cuda()
                target = target.cuda()

            with torch.no_grad():
                outputs_batch = model(img_rgb)[0]

                # print(outputs_batch)
            outputs_full.append(outputs_batch)
            labels_full.append(target)
    else:
        for (batch_sample, single_sample, label) in tqdm(iter(loader), desc="Full forward pass", total=len(loader),
                                                         disable=not verbose):

            if torch.cuda.is_available():
                single_sample = single_sample.cuda()
                label = label.cuda()

            with torch.no_grad():
                outputs_batch = model(single_sample)
                # print(outputs_batch)
            outputs_full.append(outputs_batch)
            labels_full.append(label)

    model.train(mode_saved)
    outputs_full = torch.cat(outputs_full, dim=0)
    labels_full = torch.cat(labels_full, dim=0)
    _, labels_predicted = torch.max(outputs_full.data, dim=1)
    accuracy = torch.sum(labels_full == labels_predicted).item() / float(len(labels_full))
    # print((labels_full - labels_predicted))
    accuracy = float("%.6f" % accuracy)

    if hter:
        predict_arr = np.array(labels_predicted.cpu())
        label_arr = np.array(labels_full.cpu())

        living_wrong = 0  # living -- spoofing
        living_right = 0
        spoofing_wrong = 0  # spoofing ---living
        spoofing_right = 0

        for i in range(len(predict_arr)):
            if predict_arr[i] == label_arr[i]:
                if label_arr[i] == 1:
                    living_right += 1
                else:
                    spoofing_right += 1
            else:
                # 错误
                if label_arr[i] == 1:
                    living_wrong += 1
                else:
                    spoofing_wrong += 1

        try:

            FRR = living_wrong / (living_wrong + living_right)
            APCER = living_wrong / (spoofing_right + living_wrong)
            NPCER = spoofing_wrong / (spoofing_wrong + living_right)
            ACER = (APCER + NPCER) / 2
            FAR = spoofing_wrong / (spoofing_wrong + spoofing_right)
            HTER = (FAR + FRR) / 2

            FAR = float("%.6f" % FAR)
            FRR = float("%.6f" % FRR)
            HTER = float("%.6f" % HTER)
            accuracy = float("%.6f" % accuracy)
        except Exception as e:
            print(living_right, living_wrong, spoofing_right, spoofing_wrong)
            return [accuracy, 1, 1, 1, 1, 1, 1]

        print(living_right, living_wrong, spoofing_right, spoofing_wrong)
        return [accuracy, FAR, FRR, HTER, APCER, NPCER, ACER]
    else:
        return [accuracy]


def calc_accuracy_pixel(model, loader, verbose=False, hter=False):
    """
    :param model: model network
    :param loader: torch.utils.data.DataLoader
    :param verbose: show progress bar, bool
    :return accuracy, float
    """
    mode_saved = model.training
    measure = nn.MSELoss()
    measure_loss = 0
    model.train(False)
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        model.cuda()
    for inputs, labels in tqdm(iter(loader), desc="Full forward pass", total=len(loader), disable=not verbose):
        if use_cuda:
            inputs = inputs.cuda()
            labels = labels.cuda()
        with torch.no_grad():
            outputs_batch = model(inputs)
        measure_loss += measure(outputs_batch, labels)
    model.train(mode_saved)

    return measure_loss / len(loader)


def train_multi_advsor(model, cost, optimizer, train_loader, test_loader, args):
    '''

    :param model:
    :param cost:
    :param optimizer:
    :param train_loader:
    :param test_loader:
    :param args:
    :return:
    '''

    print(args)
    criterion_absolute_loss = nn.BCELoss()
    # Initialize and open timer
    start = time.time()

    if not os.path.exists(args.model_root):
        os.makedirs(args.model_root)
    if not os.path.exists(args.log_root):
        os.makedirs(args.log_root)

    models_dir = args.model_root + '/' + args.name + '.pt'
    log_dir = args.log_root + '/' + args.name + '.csv'

    # save args
    with open(log_dir, 'a+', newline='') as f:
        my_writer = csv.writer(f)
        args_dict = vars(args)
        for key, value in args_dict.items():
            my_writer.writerow([key, value])
        f.close()

    #  learning rate decay
    if args.lr_decrease == 'cos':
        print("lrcos is using")
        cos_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.train_epoch + 20, eta_min=0)

        if args.lr_warmup:
            scheduler_warmup = GradualWarmupScheduler(args, optimizer, multiplier=1,
                                                      after_scheduler=cos_scheduler)
    elif args.lr_decrease == 'multi_step':
        print("multi_step is using")
        cos_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[np.int(args.train_epoch * 1 / 6),
                                                                              np.int(args.train_epoch * 2 / 6),
                                                                              np.int(args.train_epoch * 3 / 6)])
        if args.lr_warmup:
            scheduler_warmup = GradualWarmupScheduler(args, optimizer, multiplier=1,
                                                      after_scheduler=cos_scheduler)

    # Training initialization
    epoch_num = args.train_epoch
    log_interval = args.log_interval
    save_interval = args.save_interval
    batch_num = 0
    train_loss = 0
    epoch = 0
    accuracy_best = 0
    log_list = []  # log need to save

    if args.retrain:
        if not os.path.exists(models_dir):
            print("no trained model")
        else:
            state_read = torch.load(models_dir)
            model.load_state_dict(state_read['model_state'])
            optimizer.load_state_dict(state_read['optim_state'])
            epoch = state_read['Epoch']
            print("retaining")

    # Train
    while epoch < epoch_num:
        for batch_idx, sample_batch in enumerate(
                tqdm(train_loader, desc="Epoch {}/{}".format(epoch, epoch_num))):

            img_rgb, ir_target, depth_target, binary_target = sample_batch['image_rgb'], sample_batch['image_ir'], \
                sample_batch['image_depth'], sample_batch['binary_label']

            ir_target = torch.unsqueeze(ir_target, 1)
            depth_target = torch.unsqueeze(depth_target, 1)

            batch_num += 1
            if torch.cuda.is_available():
                img_rgb, ir_target, depth_target, binary_target = img_rgb.cuda(), ir_target.cuda(), depth_target.cuda(), binary_target.cuda()

            optimizer.zero_grad()

            batch_size = img_rgb.shape[0]
            index = binary_target.cpu()
            index = torch.unsqueeze(index, 1)
            y_one_hot = torch.zeros(batch_size, args.class_num).scatter_(1, index, 1)
            if torch.cuda.is_available():
                y_one_hot = y_one_hot.cuda()

            if args.method == 'deeppix':
                ir_out, depth_out, binary_out = model(img_rgb)
                loss1 = cost(binary_out, y_one_hot)
                loss2 = criterion_absolute_loss(ir_out, ir_target.float())
                loss3 = criterion_absolute_loss(depth_out, depth_target.float())
                loss = (loss1 + loss2 + loss3) / 3
            elif args.method == 'pyramid':
                if args.origin_deeppix:
                    out_8x8, out_4x4, out_2x2, out_1x1, binary_out = model(img_rgb)
                    loss1 = cost(binary_out, y_one_hot)
                    depth_target_8x8 = F.adaptive_avg_pool2d(depth_target, (8, 8))
                    depth_target_4x4 = F.adaptive_avg_pool2d(depth_target, (4, 4))
                    depth_target_2x2 = F.adaptive_avg_pool2d(depth_target, (2, 2))
                    depth_target_1x1 = F.adaptive_avg_pool2d(depth_target, (1, 1))

                    ir_target_8x8 = F.adaptive_avg_pool2d(ir_target, (8, 8))
                    ir_target_4x4 = F.adaptive_avg_pool2d(ir_target, (4, 4))
                    ir_target_2x2 = F.adaptive_avg_pool2d(ir_target, (2, 2))
                    ir_target_1x1 = F.adaptive_avg_pool2d(ir_target, (1, 1))

                    loss2 = criterion_absolute_loss(out_8x8, depth_target_8x8.float())
                    loss3 = criterion_absolute_loss(out_4x4, depth_target_4x4.float())
                    loss4 = criterion_absolute_loss(out_2x2, depth_target_2x2.float())
                    loss5 = criterion_absolute_loss(out_1x1, depth_target_1x1.float())

                    loss = (loss1 + loss2 + loss3 + loss4 + loss5) / 5
                else:
                    out_depth_32x32, out_depth_16x16, out_ir_32x32, out_ir_16x16, binary_out = model(img_rgb)
                    loss1 = cost(binary_out, y_one_hot)

                    depth_target_32x32 = F.adaptive_avg_pool2d(depth_target, (32, 32))
                    depth_target_16x16 = F.adaptive_avg_pool2d(depth_target, (16, 16))

                    ir_target_32x32 = F.adaptive_avg_pool2d(ir_target, (32, 32))
                    ir_target_16x16 = F.adaptive_avg_pool2d(ir_target, (16, 16))

                    loss2 = criterion_absolute_loss(out_depth_32x32, depth_target_32x32.float())
                    loss3 = criterion_absolute_loss(out_ir_32x32, ir_target_32x32.float())
                    loss4 = criterion_absolute_loss(out_depth_16x16, depth_target_16x16.float())
                    loss5 = criterion_absolute_loss(out_ir_16x16, ir_target_16x16.float())
                    loss = (loss1 + loss2 + loss3 + loss4 + loss5) / 5

            else:
                print("loss error")
                loss = torch.tensor(0)

            train_loss += loss.item()
            loss.backward()
            optimizer.step()
            # if batch_idx % log_interval == 0:  # 准备打印相关信息，args.log_interval是最开头设置的好了的参数
            #     print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            #         epoch, batch_idx * len(data), len(train_loader.dataset),
            #                100. * batch_idx / len(train_loader), loss.item()))

        # testing
        result_test = calc_accuracy_multi_advisor(model, loader=test_loader, args=args)
        accuracy_test = result_test[0]
        if accuracy_test > accuracy_best and epoch > 12:
            accuracy_best = accuracy_test
            save_path = os.path.join(args.model_root, args.name + '.pth')
            torch.save(model.state_dict(), save_path, )
        log_list.append(train_loss / len(train_loader))
        log_list.append(accuracy_test)
        log_list.append(accuracy_best)
        log_list.append(torch.detach(loss1).cpu().numpy())
        log_list.append(torch.detach(loss2).cpu().numpy())
        log_list.append(torch.detach(loss3).cpu().numpy())

        print(
            "Epoch {}, loss={:.5f}, accuracy_test={:.5f},  accuracy_best={:.5f}".format(epoch,
                                                                                        train_loss / len(train_loader),
                                                                                        accuracy_test, accuracy_best))
        train_loss = 0

        print(loss1.cpu(), loss2.cpu(), loss3.cpu())
        if args.lr_decrease:
            if args.lr_warmup:
                scheduler_warmup.step(epoch=epoch)
            else:
                cos_scheduler.step(epoch=epoch)
        if epoch < 20:
            print(epoch, optimizer.param_groups[0]['lr'])

        # save model and para
        if epoch % save_interval == 0:
            train_state = {
                "Epoch": epoch,
                "model_state": model.state_dict(),
                "optim_state": optimizer.state_dict(),
                "args": args
            }
            models_dir = args.model_root + '/' + args.name + '.pt'
            torch.save(train_state, models_dir)

        #  save log
        with open(log_dir, 'a+', newline='') as f:
            # 训练结果
            my_writer = csv.writer(f)
            my_writer.writerow(log_list)
            log_list = []
        epoch = epoch + 1
    train_duration_sec = int(time.time() - start)
    print("training is end", train_duration_sec)


def train_base_multi_origin(model, cost, optimizer, train_loader, test_loader, args):
    '''
    适用于多模态分类的基础训练函数
    :param model:
    :param cost:
    :param optimizer:
    :param train_loader:
    :param test_loader:
    :param args:
    :return:
    '''
    print(args)

    # Initialize and open timer
    start = time.time()

    if not os.path.exists(args.model_root):
        os.makedirs(args.model_root)
    if not os.path.exists(args.log_root):
        os.makedirs(args.log_root)

    models_dir = args.model_root + '/' + args.name + '.pt'
    log_dir = args.log_root + '/' + args.name + '.csv'

    # save args
    with open(log_dir, 'a+', newline='') as f:
        my_writer = csv.writer(f)
        args_dict = vars(args)
        for key, value in args_dict.items():
            my_writer.writerow([key, value])
        f.close()

    #  learning rate decay
    if args.lr_decrease == 'cos':
        print("lrcos is using")
        cos_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.train_epoch + 20, eta_min=0)

        if args.lr_warmup:
            scheduler_warmup = GradualWarmupScheduler(args, optimizer, multiplier=1,
                                                      after_scheduler=cos_scheduler)
    elif args.lr_decrease == 'multi_step':
        print("multi_step is using")
        cos_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[int(args.train_epoch * 1 / 6),
                                                                              int(args.train_epoch * 2 / 6),
                                                                              int(args.train_epoch * 3 / 6)])
        if args.lr_warmup:
            scheduler_warmup = GradualWarmupScheduler(args, optimizer, multiplier=1,
                                                      after_scheduler=cos_scheduler)
    else:
        if args.lr_warmup:
            scheduler_warmup = GradualWarmupScheduler(args, optimizer, multiplier=1)

    # Training initialization
    epoch_num = args.train_epoch
    log_interval = args.log_interval
    save_interval = args.save_interval
    batch_num = 0
    train_loss = 0
    epoch = 0
    accuracy_best = 0
    acer_best = 1.0
    hter_best = 1.0
    loss_kl_sum = 0
    log_list = []  # log need to save

    if args.retrain:
        if not os.path.exists(models_dir):
            print("no trained model")
        else:
            state_read = torch.load(models_dir)
            model.load_state_dict(state_read['model_state'])
            optimizer.load_state_dict(state_read['optim_state'])
            epoch = state_read['Epoch']
            print("retaining")

    # Train
    while epoch < epoch_num:
        for batch_idx, batch_sample in enumerate(
                tqdm(train_loader, desc="Epoch {}/{}".format(epoch, epoch_num))):

            batch_num += 1
            img_rgb, img_ir, img_depth, target = batch_sample['image_x'], batch_sample['image_ir'], \
                batch_sample['image_depth'], batch_sample[
                'binary_label']
            if epoch == 0:
                continue
            if torch.cuda.is_available():
                img_rgb = img_rgb.cuda()
                img_ir = img_ir.cuda()
                img_depth = img_depth.cuda()
                target = target.cuda()

            # optimizer.zero_grad()
            for p in model.parameters():
                p.grad = None

            model.args.epoch = epoch
            output = model(img_rgb, img_ir, img_depth)

            if isinstance(output, tuple):
                output = output[0]
            loss_cls = cost(output, target)
            loss = loss_cls

            train_loss += loss.item()

            loss.backward()

            # if batch_num>10:
            #     print("weight.grad:", model.special_bone_rgb[0].weight.grad.mean(), model.special_bone_rgb[0].weight.grad.min(), model.special_bone_rgb[0].weight.grad.max())
            #     print("weight.grad:", model.special_bone_ir[0].weight.grad.mean(), model.special_bone_ir[0].weight.grad.min(), model.special_bone_ir[0].weight.grad.max())
            #     print("weight.grad:", model.special_bone_depth[0].weight.grad.mean(), model.special_bone_depth[0].weight.grad.min(), model.special_bone_depth[0].weight.grad.max())

            if (model.special_bone_rgb[0].weight.grad is None) or (model.special_bone_ir[0].weight.grad is None) or (
                    model.special_bone_depth[0].weight.grad is None):
                print("none!!!!!!none!!!!!")

            # print(model.special_bone_rgb[0].weight.grad)
            # print(model.special_bone_ir[0].weight.grad)
            # print(model.special_bone_depth[0].weight.grad)

            optimizer.step()

        # testing
        result_test, _ = calc_accuracy_multi(model, loader=test_loader, hter=True, verbose=True)
        hter_test = result_test[3]
        acer_test = result_test[-1]
        # save_path = os.path.join(args.model_root, args.name + "_epoch_" + str(epoch)
        #                          + '.pth')
        # torch.save(model.state_dict(), save_path)

        accuracy_test = result_test[0]

        if acer_test < acer_best and epoch > 15:
            acer_best = acer_test
            save_path = os.path.join(args.model_root, args.name + '_acer_best_' + '.pth')
            torch.save(model.state_dict(), save_path, )

        if hter_test < hter_best and epoch > 15:
            hter_best = hter_test
            save_path = os.path.join(args.model_root, args.name + '_hter_best_' + '.pth')
            torch.save(model.state_dict(), save_path, )

        if accuracy_test > accuracy_best and epoch > 15:
            accuracy_best = accuracy_test
            save_path = os.path.join(args.model_root, args.name + '.pth')
            torch.save(model.state_dict(), save_path, )

        log_list.append(train_loss / len(train_loader))
        log_list.append(accuracy_test)
        log_list.append(accuracy_best)
        print(
            "Epoch {}, loss={:.5f}, accuracy_test={:.5f},  accuracy_best={:.5f}".format(epoch,
                                                                                        train_loss / len(train_loader),
                                                                                        accuracy_test, accuracy_best))
        train_loss = 0

        print(loss_kl_sum / len(train_loader))

        loss_kl_sum = 0

        if args.lr_decrease:
            if args.lr_warmup:
                scheduler_warmup.step(epoch=epoch)
            else:
                cos_scheduler.step(epoch=epoch)
        if epoch < 20:
            print(epoch, optimizer.param_groups[0]['lr'])

        # save model and para
        if epoch % save_interval == 0:
            train_state = {
                "Epoch": epoch,
                "model_state": model.state_dict(),
                "optim_state": optimizer.state_dict(),
                "args": args
            }
            models_dir = args.model_root + '/' + args.name + '.pt'
            torch.save(train_state, models_dir)

        #  save log
        with open(log_dir, 'a+', newline='') as f:
            # 训练结果
            my_writer = csv.writer(f)
            my_writer.writerow(log_list)
            log_list = []
        epoch = epoch + 1
    train_duration_sec = int(time.time() - start)
    print("training is end", train_duration_sec)


def train_base_multi_baseline(model, cost, optimizer, train_loader, test_loader, args):
    '''
    适用于多模态分类的基础训练函数
    :param model:
    :param cost:
    :param optimizer:
    :param train_loader:
    :param test_loader:
    :param args:
    :return:
    '''
    print(args)

    # Initialize and open timer
    start = time.time()

    if not os.path.exists(args.model_root):
        os.makedirs(args.model_root)
    if not os.path.exists(args.log_root):
        os.makedirs(args.log_root)

    models_dir = args.model_root + '/' + args.name + '.pt'
    log_dir = args.log_root + '/' + args.name + '.csv'

    # save args
    with open(log_dir, 'a+', newline='') as f:
        my_writer = csv.writer(f)
        args_dict = vars(args)
        for key, value in args_dict.items():
            my_writer.writerow([key, value])
        f.close()

    #  learning rate decay
    if args.lr_decrease == 'cos':
        print("lrcos is using")
        cos_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.train_epoch + 20, eta_min=0)

        if args.lr_warmup:
            scheduler_warmup = GradualWarmupScheduler(args, optimizer, multiplier=1,
                                                      after_scheduler=cos_scheduler)
    elif args.lr_decrease == 'multi_step':
        print("multi_step is using")
        cos_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[int(args.train_epoch * 1 / 6),
                                                                              int(args.train_epoch * 2 / 6),
                                                                              int(args.train_epoch * 3 / 6)])
        if args.lr_warmup:
            scheduler_warmup = GradualWarmupScheduler(args, optimizer, multiplier=1,
                                                      after_scheduler=cos_scheduler)
    else:
        if args.lr_warmup:
            scheduler_warmup = GradualWarmupScheduler(args, optimizer, multiplier=1)

    # Training initialization
    epoch_num = args.train_epoch
    log_interval = args.log_interval
    save_interval = args.save_interval
    batch_num = 0
    train_loss = 0
    epoch = 0
    accuracy_best = 0
    acer_best = 1.0
    hter_best = 1.0
    loss_kl_sum = 0
    log_list = []  # log need to save
    varaince_list = [0, 0, 0, 0, 0, 0, 0]

    modality_combination = torch.tensor(
        [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1], [0, 1, 1], [1, 1, 1]]).float()

    if args.retrain:
        if not os.path.exists(models_dir):
            print("no trained model")
        else:
            state_read = torch.load(models_dir)
            model.load_state_dict(state_read['model_state'])
            optimizer.load_state_dict(state_read['optim_state'])
            epoch = state_read['Epoch']
            print("retaining")

    # Train
    while epoch < epoch_num:
        for batch_idx, batch_sample in enumerate(
                tqdm(train_loader, desc="Epoch {}/{}".format(epoch, epoch_num))):

            batch_num += 1
            img_rgb, img_ir, img_depth, target = batch_sample['image_x'], batch_sample['image_ir'], \
                batch_sample['image_depth'], batch_sample[
                'binary_label']
            # if epoch == 0:
            #     continue
            if torch.cuda.is_available():
                img_rgb = img_rgb.cuda()
                img_ir = img_ir.cuda()
                img_depth = img_depth.cuda()
                target = target.cuda()

            # optimizer.zero_grad()
            for p in model.parameters():
                p.grad = None

            model.args.epoch = epoch
            output = model(img_rgb, img_ir, img_depth)

            if isinstance(output, tuple):
                output = output[0]
            loss_cls = cost(output, target)

            loss = loss_cls

            train_loss += loss.item()

            loss.backward()

            optimizer.step()

        # testing
        result_test, _ = calc_accuracy_multi(model, loader=test_loader, hter=True, verbose=True)
        hter_test = result_test[3]
        acer_test = result_test[-1]

        print(np.array(varaince_list) / len(train_loader))
        varaince_list = [0, 0, 0, 0, 0, 0, 0]
        # save_path = os.path.join(args.model_root, args.name + "_epoch_" + str(epoch)
        #                          + '.pth')
        # torch.save(model.state_dict(), save_path)

        accuracy_test = result_test[0]

        if acer_test < acer_best and epoch > 15:
            acer_best = acer_test
            save_path = os.path.join(args.model_root, args.name + '_acer_best_' + '.pth')
            torch.save(model.state_dict(), save_path, )

        if hter_test < hter_best and epoch > 15:
            hter_best = hter_test
            save_path = os.path.join(args.model_root, args.name + '_hter_best_' + '.pth')
            torch.save(model.state_dict(), save_path, )

        if accuracy_test > accuracy_best and epoch > 15:
            accuracy_best = accuracy_test
            save_path = os.path.join(args.model_root, args.name + '.pth')
            torch.save(model.state_dict(), save_path, )

        log_list.append(train_loss / len(train_loader))
        log_list.append(accuracy_test)
        log_list.append(accuracy_best)
        print(
            "Epoch {}, loss={:.5f}, accuracy_test={:.5f},  accuracy_best={:.5f}".format(epoch,
                                                                                        train_loss / len(train_loader),
                                                                                        accuracy_test, accuracy_best))
        train_loss = 0

        if args.lr_decrease:
            if args.lr_warmup:
                scheduler_warmup.step(epoch=epoch)
            else:
                cos_scheduler.step(epoch=epoch)
        if epoch < 20:
            print(epoch, optimizer.param_groups[0]['lr'])

        # save model and para
        if epoch % save_interval == 0:
            train_state = {
                "Epoch": epoch,
                "model_state": model.state_dict(),
                "optim_state": optimizer.state_dict(),
                "args": args
            }
            models_dir = args.model_root + '/' + args.name + '.pt'
            torch.save(train_state, models_dir)

        #  save log
        with open(log_dir, 'a+', newline='') as f:
            # 训练结果
            my_writer = csv.writer(f)
            my_writer.writerow(log_list)
            log_list = []
        epoch = epoch + 1
    train_duration_sec = int(time.time() - start)
    print("training is end", train_duration_sec)


def train_base_multi(model, cost, optimizer, train_loader, test_loader, args):
    '''
    适用于多模态分类的基础训练函数
    :param model:
    :param cost:
    :param optimizer:
    :param train_loader:
    :param test_loader:
    :param args:
    :return:
    '''
    print(args)

    # Initialize and open timer
    start = time.time()

    if not os.path.exists(args.model_root):
        os.makedirs(args.model_root)
    if not os.path.exists(args.log_root):
        os.makedirs(args.log_root)

    models_dir = args.model_root + '/' + args.name + '.pt'
    log_dir = args.log_root + '/' + args.name + '.csv'

    # save args
    with open(log_dir, 'a+', newline='') as f:
        my_writer = csv.writer(f)
        args_dict = vars(args)
        for key, value in args_dict.items():
            my_writer.writerow([key, value])
        f.close()

    #  learning rate decay
    if args.lr_decrease == 'cos':
        print("lrcos is using")
        cos_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.train_epoch + 20, eta_min=0)

        if args.lr_warmup:
            scheduler_warmup = GradualWarmupScheduler(args, optimizer, multiplier=1,
                                                      after_scheduler=cos_scheduler)
    elif args.lr_decrease == 'multi_step':
        print("multi_step is using")
        cos_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[int(args.train_epoch * 1 / 6),
                                                                              int(args.train_epoch * 2 / 6),
                                                                              int(args.train_epoch * 3 / 6)])
        if args.lr_warmup:
            scheduler_warmup = GradualWarmupScheduler(args, optimizer, multiplier=1,
                                                      after_scheduler=cos_scheduler)
    else:
        if args.lr_warmup:
            scheduler_warmup = GradualWarmupScheduler(args, optimizer, multiplier=1)

    # Training initialization
    epoch_num = args.train_epoch
    log_interval = args.log_interval
    save_interval = args.save_interval
    batch_num = 0
    train_loss = 0
    epoch = 0
    accuracy_best = 0
    acer_best = 1.0
    hter_best = 1.0
    loss_kl_sum = 0
    log_list = []  # log need to save
    varaince_list = [0, 0, 0, 0, 0, 0, 0]

    modality_combination = torch.tensor(
        [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1], [0, 1, 1], [1, 1, 1]]).float()

    if args.retrain:
        if not os.path.exists(models_dir):
            print("no trained model")
        else:
            state_read = torch.load(models_dir)
            model.load_state_dict(state_read['model_state'])
            optimizer.load_state_dict(state_read['optim_state'])
            epoch = state_read['Epoch']
            print("retaining")

    # Train
    while epoch < epoch_num:
        for batch_idx, batch_sample in enumerate(
                tqdm(train_loader, desc="Epoch {}/{}".format(epoch, epoch_num))):

            batch_num += 1
            img_rgb, img_ir, img_depth, target = batch_sample['image_x'], batch_sample['image_ir'], \
                batch_sample['image_depth'], batch_sample[
                'binary_label']
            # if epoch == 0:
            #     continue
            if torch.cuda.is_available():
                img_rgb = img_rgb.cuda()
                img_ir = img_ir.cuda()
                img_depth = img_depth.cuda()
                target = target.cuda()

            # optimizer.zero_grad()
            for p in model.parameters():
                p.grad = None

            model.args.epoch = epoch
            output = model(img_rgb, img_ir, img_depth)

            (mul, std) = output[-1]
            p = output[1]

            if isinstance(output, tuple):
                output = output[0]
            loss_cls = cost(output, target)
            if torch.sum(std) == 0:  # 正常训练。没有分布化
                loss = loss_cls
                loss_kl_sum += 0
            else:

                variance_dul = std ** 2
                variance_dul = variance_dul.view(variance_dul.shape[0], -1)
                mul = mul.view(mul.shape[0], -1)
                loss_kl = torch.sum(((variance_dul + mul ** 2 - torch.log(variance_dul) - 1) * 0.5), dim=1)
                loss_kl = torch.mean(loss_kl)

                p = p.cpu().detach()
                p = p.view(p.shape[0], -1)
                variance_dul = torch.mean(variance_dul, 1)
                for i in range(len(modality_combination)):
                    # print(p.shape)
                    index = (p == modality_combination[i])
                    index = index[:, 0] & index[:, 1] & index[:, 2]
                    # print(index)
                    # print(index.shape)
                    varaince_slect = variance_dul[index]
                    varaince_list[i] += (torch.mean(varaince_slect)).cpu().detach().numpy()

                if epoch > 5:
                    loss = loss_cls + args.kl_scale * loss_kl
                else:
                    loss = loss_cls

                loss_kl_sum += loss_kl.item()

            train_loss += loss.item()

            loss.backward()

            # if batch_num>10:
            #     print("weight.grad:", model.special_bone_rgb[0].weight.grad.mean(), model.special_bone_rgb[0].weight.grad.min(), model.special_bone_rgb[0].weight.grad.max())
            #     print("weight.grad:", model.special_bone_ir[0].weight.grad.mean(), model.special_bone_ir[0].weight.grad.min(), model.special_bone_ir[0].weight.grad.max())
            #     print("weight.grad:", model.special_bone_depth[0].weight.grad.mean(), model.special_bone_depth[0].weight.grad.min(), model.special_bone_depth[0].weight.grad.max())

            if (model.special_bone_rgb[0].weight.grad is None) or (model.special_bone_ir[0].weight.grad is None) or (
                    model.special_bone_depth[0].weight.grad is None):
                print("none!!!!!!none!!!!!")

            # print(model.special_bone_rgb[0].weight.grad)
            # print(model.special_bone_ir[0].weight.grad)
            # print(model.special_bone_depth[0].weight.grad)

            optimizer.step()

        # testing
        result_test, _ = calc_accuracy_multi(model, loader=test_loader, hter=True, verbose=True)
        hter_test = result_test[3]
        acer_test = result_test[-1]

        print(np.array(varaince_list) / len(train_loader))
        varaince_list = [0, 0, 0, 0, 0, 0, 0]
        # save_path = os.path.join(args.model_root, args.name + "_epoch_" + str(epoch)
        #                          + '.pth')
        # torch.save(model.state_dict(), save_path)

        accuracy_test = result_test[0]

        if acer_test < acer_best and epoch > 15:
            acer_best = acer_test
            save_path = os.path.join(args.model_root, args.name + '_acer_best_' + '.pth')
            torch.save(model.state_dict(), save_path, )

        if hter_test < hter_best and epoch > 15:
            hter_best = hter_test
            save_path = os.path.join(args.model_root, args.name + '_hter_best_' + '.pth')
            torch.save(model.state_dict(), save_path, )

        if accuracy_test > accuracy_best and epoch > 15:
            accuracy_best = accuracy_test
            save_path = os.path.join(args.model_root, args.name + '.pth')
            torch.save(model.state_dict(), save_path, )

        log_list.append(train_loss / len(train_loader))
        log_list.append(accuracy_test)
        log_list.append(accuracy_best)
        print(
            "Epoch {}, loss={:.5f}, accuracy_test={:.5f},  accuracy_best={:.5f}".format(epoch,
                                                                                        train_loss / len(train_loader),
                                                                                        accuracy_test, accuracy_best))
        train_loss = 0

        print(loss_kl_sum / len(train_loader))

        loss_kl_sum = 0

        if args.lr_decrease:
            if args.lr_warmup:
                scheduler_warmup.step(epoch=epoch)
            else:
                cos_scheduler.step(epoch=epoch)
        if epoch < 20:
            print(epoch, optimizer.param_groups[0]['lr'])

        # save model and para
        if epoch % save_interval == 0:
            train_state = {
                "Epoch": epoch,
                "model_state": model.state_dict(),
                "optim_state": optimizer.state_dict(),
                "args": args
            }
            models_dir = args.model_root + '/' + args.name + '.pt'
            torch.save(train_state, models_dir)

        #  save log
        with open(log_dir, 'a+', newline='') as f:
            # 训练结果
            my_writer = csv.writer(f)
            my_writer.writerow(log_list)
            log_list = []
        epoch = epoch + 1
    train_duration_sec = int(time.time() - start)
    print("training is end", train_duration_sec)


def train_base_multi_shaspec(model, cost, optimizer, train_loader, test_loader, args):
    '''
    适用于多模态分类的基础训练函数
    :param model:
    :param cost:
    :param optimizer:
    :param train_loader:
    :param test_loader:
    :param args:
    :return:
    '''
    print(args)

    # Initialize and open timer
    start = time.time()

    if not os.path.exists(args.model_root):
        os.makedirs(args.model_root)
    if not os.path.exists(args.log_root):
        os.makedirs(args.log_root)

    models_dir = args.model_root + '/' + args.name + '.pt'
    log_dir = args.log_root + '/' + args.name + '.csv'

    mse_func = nn.MSELoss()

    # save args
    with open(log_dir, 'a+', newline='') as f:
        my_writer = csv.writer(f)
        args_dict = vars(args)
        for key, value in args_dict.items():
            my_writer.writerow([key, value])
        f.close()

    #  learning rate decay
    if args.lr_decrease == 'cos':
        print("lrcos is using")
        cos_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.train_epoch + 20, eta_min=0)

        if args.lr_warmup:
            scheduler_warmup = GradualWarmupScheduler(args, optimizer, multiplier=1,
                                                      after_scheduler=cos_scheduler)
    elif args.lr_decrease == 'multi_step':
        print("multi_step is using")
        cos_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[int(args.train_epoch * 1 / 6),
                                                                              int(args.train_epoch * 2 / 6),
                                                                              int(args.train_epoch * 3 / 6)])
        if args.lr_warmup:
            scheduler_warmup = GradualWarmupScheduler(args, optimizer, multiplier=1,
                                                      after_scheduler=cos_scheduler)
    else:
        if args.lr_warmup:
            scheduler_warmup = GradualWarmupScheduler(args, optimizer, multiplier=1)

    # Training initialization
    epoch_num = args.train_epoch
    log_interval = args.log_interval
    save_interval = args.save_interval
    batch_num = 0
    train_loss = 0
    epoch = 0
    accuracy_best = 0
    acer_best = 1.0
    hter_best = 1.0
    loss_kl_sum = 0
    log_list = []  # log need to save
    varaince_list = [0, 0, 0, 0, 0, 0, 0]
    cls_sum = 0
    dco_loss_sum = 0
    dao_loss_sum = 0
    unimodal_loss_sum = 0
    modality_combination = torch.tensor(
        [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1], [0, 1, 1], [1, 1, 1]]).float()

    if args.retrain:
        if not os.path.exists(models_dir):
            print("no trained model")
        else:
            state_read = torch.load(models_dir)
            model.load_state_dict(state_read['model_state'])
            optimizer.load_state_dict(state_read['optim_state'])
            epoch = state_read['Epoch']
            print("retaining")

    # Train
    while epoch < epoch_num:
        for batch_idx, batch_sample in enumerate(
                tqdm(train_loader, desc="Epoch {}/{}".format(epoch, epoch_num))):

            batch_num += 1
            img_rgb, img_ir, img_depth, target = batch_sample['image_x'], batch_sample['image_ir'], \
                batch_sample['image_depth'], batch_sample[
                'binary_label']
            # if epoch == 0:
            #     continue
            if torch.cuda.is_available():
                img_rgb = img_rgb.cuda()
                img_ir = img_ir.cuda()
                img_depth = img_depth.cuda()
                target = target.cuda()

            # optimizer.zero_grad()
            for p in model.parameters():
                p.grad = None

            model.args.epoch = epoch
            target_predict, dco_predict, dco_label, m1_feature_share_cache, m2_feature_share_cache, m3_feature_share_cache, fusion_feature, m1_predict, m2_predict, m3_predict = model(
                img_rgb, img_ir, img_depth)

            task_loss = cost(target_predict, target)
            cls_loss2 = cost(m1_predict, target)
            cls_loss3 = cost(m2_predict, target)
            cls_loss4 = cost(m3_predict, target)

            unimodal_loss = cls_loss2 + cls_loss3 + cls_loss4

            dao_loss = mse_func(m1_feature_share_cache, m2_feature_share_cache) + mse_func(m2_feature_share_cache,
                                                                                           m3_feature_share_cache)

            dco_loss = cost(dco_predict, dco_label)

            loss = task_loss + 1.0 * dao_loss + 0.02 * dco_loss + unimodal_loss * 1

            train_loss += loss.item()

            loss.backward()

            cls_sum += task_loss.item()
            dao_loss_sum += dao_loss.item()
            dco_loss_sum += dco_loss.item()
            unimodal_loss_sum += unimodal_loss.item()

            optimizer.step()

        # testing
        result_test, _ = calc_accuracy_multi(model, loader=test_loader, hter=True, verbose=True)
        hter_test = result_test[3]
        acer_test = result_test[-1]

        print(np.array(varaince_list) / len(train_loader))
        varaince_list = [0, 0, 0, 0, 0, 0, 0]
        # save_path = os.path.join(args.model_root, args.name + "_epoch_" + str(epoch)
        #                          + '.pth')
        # torch.save(model.state_dict(), save_path)

        accuracy_test = result_test[0]

        if acer_test < acer_best and epoch > 15:
            acer_best = acer_test
            save_path = os.path.join(args.model_root, args.name + '_acer_best_' + '.pth')
            torch.save(model.state_dict(), save_path, )

        if hter_test < hter_best and epoch > 15:
            hter_best = hter_test
            save_path = os.path.join(args.model_root, args.name + '_hter_best_' + '.pth')
            torch.save(model.state_dict(), save_path, )

        if accuracy_test > accuracy_best and epoch > 15:
            accuracy_best = accuracy_test
            save_path = os.path.join(args.model_root, args.name + '.pth')
            torch.save(model.state_dict(), save_path, )

        log_list.append(train_loss / len(train_loader))
        log_list.append(accuracy_test)
        log_list.append(accuracy_best)
        print(
            "Epoch {}, loss={:.5f}, accuracy_test={:.5f},  accuracy_best={:.5f}".format(epoch,
                                                                                        train_loss / len(train_loader),
                                                                                        accuracy_test, accuracy_best))
        train_loss = 0

        print(cls_sum / len(train_loader), dao_loss_sum / len(train_loader), dco_loss_sum / len(train_loader),
              unimodal_loss_sum / len(train_loader))
        cls_sum = 0
        dco_loss_sum = 0
        dao_loss_sum = 0
        unimodal_loss_sum = 0

        loss_kl_sum = 0

        if args.lr_decrease:
            if args.lr_warmup:
                scheduler_warmup.step(epoch=epoch)
            else:
                cos_scheduler.step(epoch=epoch)
        if epoch < 20:
            print(epoch, optimizer.param_groups[0]['lr'])

        # save model and para
        if epoch % save_interval == 0:
            train_state = {
                "Epoch": epoch,
                "model_state": model.state_dict(),
                "optim_state": optimizer.state_dict(),
                "args": args
            }
            models_dir = args.model_root + '/' + args.name + '.pt'
            torch.save(train_state, models_dir)

        #  save log
        with open(log_dir, 'a+', newline='') as f:
            # 训练结果
            my_writer = csv.writer(f)
            my_writer.writerow(log_list)
            log_list = []
        epoch = epoch + 1
    train_duration_sec = int(time.time() - start)
    print("training is end", train_duration_sec)


from random import shuffle


def train_base_multi_mix(model, cost, optimizer, train_loader, test_loader, args):
    '''
    适用于多模态分类的基础训练函数
    :param model:
    :param cost:
    :param optimizer:
    :param train_loader:
    :param test_loader:
    :param args:
    :return:
    '''
    print(args)

    # Initialize and open timer
    start = time.time()

    if not os.path.exists(args.model_root):
        os.makedirs(args.model_root)
    if not os.path.exists(args.log_root):
        os.makedirs(args.log_root)

    models_dir = args.model_root + '/' + args.name + '.pt'
    log_dir = args.log_root + '/' + args.name + '.csv'

    # save args
    with open(log_dir, 'a+', newline='') as f:
        my_writer = csv.writer(f)
        args_dict = vars(args)
        for key, value in args_dict.items():
            my_writer.writerow([key, value])
        f.close()

    #  learning rate decay
    if args.lr_decrease == 'cos':
        print("lrcos is using")
        cos_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.train_epoch + 20, eta_min=0)

        if args.lr_warmup:
            scheduler_warmup = GradualWarmupScheduler(args, optimizer, multiplier=1,
                                                      after_scheduler=cos_scheduler)
    elif args.lr_decrease == 'multi_step':
        print("multi_step is using")
        cos_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[int(args.train_epoch * 1 / 6),
                                                                              int(args.train_epoch * 2 / 6),
                                                                              int(args.train_epoch * 3 / 6)])
        if args.lr_warmup:
            scheduler_warmup = GradualWarmupScheduler(args, optimizer, multiplier=1,
                                                      after_scheduler=cos_scheduler)
    else:
        if args.lr_warmup:
            scheduler_warmup = GradualWarmupScheduler(args, optimizer, multiplier=1)

    # Training initialization
    epoch_num = args.train_epoch
    log_interval = args.log_interval
    save_interval = args.save_interval
    batch_num = 0
    train_loss = 0
    epoch = 0
    accuracy_best = 0
    acer_best = 1.0
    hter_best = 1.0
    loss_kl_sum = 0
    log_list = []  # log need to save

    if args.retrain:
        if not os.path.exists(models_dir):
            print("no trained model")
        else:
            state_read = torch.load(models_dir)
            model.load_state_dict(state_read['model_state'])
            optimizer.load_state_dict(state_read['optim_state'])
            epoch = state_read['Epoch']
            print("retaining")

    # Train
    while epoch < epoch_num:
        for batch_idx, batch_sample in enumerate(
                tqdm(train_loader, desc="Epoch {}/{}".format(epoch, epoch_num))):

            batch_num += 1
            img_rgb, img_ir, img_depth, target = batch_sample['image_x'], batch_sample['image_ir'], \
                batch_sample['image_depth'], batch_sample[
                'binary_label']
            if epoch == 1:
                continue
            if torch.cuda.is_available():
                img_rgb = img_rgb.cuda()
                img_ir = img_ir.cuda()
                img_depth = img_depth.cuda()
                target = target.cuda()

            # optimizer.zero_grad()
            for p in model.parameters():
                p.grad = None

            model.args.epoch = epoch
            output = model(img_rgb, img_ir, img_depth)

            (mul, std) = output[-1]

            if isinstance(output, tuple):
                output = output[0]

            mul = mul.view(mul.shape[0], -1)
            mul = torch.mean(mul, dim=1)
            std = std.view(std.shape[0], -1)
            std = torch.mean(std, dim=1)

            c = list(zip(output[:, 0], output[:, 1], target, mul, std))
            shuffle(c)
            shuffle_output1, shuffle_output_2, shuffle_target, shuffle_mul, shuffle_std = zip(*c)
            shuffle_output1 = torch.tensor(shuffle_output1).cuda().unsqueeze(dim=1)
            shuffle_output_2 = torch.tensor(shuffle_output_2).cuda().unsqueeze(dim=1)
            # print(shuffle_output1)

            shuffle_output = torch.cat((shuffle_output1, shuffle_output_2), dim=1)

            # print(shuffle_output)
            shuffle_target = torch.tensor(shuffle_target).cuda()
            shuffle_mul = torch.tensor(shuffle_mul).cuda()
            shuffle_std = torch.tensor(shuffle_std).cuda()
            # shuffle_output=[torch.tensor(data) for data in shuffle_output]

            std1 = std / (shuffle_std + std)
            std2 = shuffle_std / (std + shuffle_std)
            std1 = std1.unsqueeze(dim=1)
            std2 = std2.unsqueeze(dim=1)
            # print(std1.shape)
            # print(std2.shape)
            # print(std1)
            # print(std2)

            output_mix = std1 * output + std2 * shuffle_output

            loss_cls_1 = cost(output_mix, target)

            loss_cls_2 = cost(output_mix, shuffle_target)

            loss_cls = loss_cls_1 + loss_cls_2

            if torch.sum(std) == 0:  # 正常训练。没有分布化
                loss = loss_cls
                loss_kl_sum += 0
            else:

                variance_dul = std ** 2
                variance_dul = variance_dul.view(variance_dul.shape[0], -1)
                mul = mul.view(mul.shape[0], -1)
                loss_kl = torch.sum(((variance_dul + mul ** 2 - torch.log(variance_dul) - 1) * 0.5), dim=1)
                loss_kl = torch.mean(loss_kl)
                loss = loss_cls + args.kl_scale * loss_kl
                loss_kl_sum += loss_kl.item()

            train_loss += loss.item()

            loss.backward()

            # if batch_num>10:
            #     print("weight.grad:", model.special_bone_rgb[0].weight.grad.mean(), model.special_bone_rgb[0].weight.grad.min(), model.special_bone_rgb[0].weight.grad.max())
            #     print("weight.grad:", model.special_bone_ir[0].weight.grad.mean(), model.special_bone_ir[0].weight.grad.min(), model.special_bone_ir[0].weight.grad.max())
            #     print("weight.grad:", model.special_bone_depth[0].weight.grad.mean(), model.special_bone_depth[0].weight.grad.min(), model.special_bone_depth[0].weight.grad.max())

            if (model.special_bone_rgb[0].weight.grad is None) or (model.special_bone_ir[0].weight.grad is None) or (
                    model.special_bone_depth[0].weight.grad is None):
                print("none!!!!!!none!!!!!")

            # print(model.special_bone_rgb[0].weight.grad)
            # print(model.special_bone_ir[0].weight.grad)
            # print(model.special_bone_depth[0].weight.grad)

            optimizer.step()

        # testing
        result_test, _ = calc_accuracy_multi(model, loader=test_loader, hter=True, verbose=True)
        hter_test = result_test[3]
        acer_test = result_test[-1]
        # save_path = os.path.join(args.model_root, args.name + "_epoch_" + str(epoch)
        #                          + '.pth')
        # torch.save(model.state_dict(), save_path)

        accuracy_test = result_test[0]

        if acer_test < acer_best and epoch > 15:
            acer_best = acer_test
            save_path = os.path.join(args.model_root, args.name + '_acer_best_' + '.pth')
            torch.save(model.state_dict(), save_path, )

        if hter_test < hter_best and epoch > 15:
            hter_best = hter_test
            save_path = os.path.join(args.model_root, args.name + '_hter_best_' + '.pth')
            torch.save(model.state_dict(), save_path, )

        if accuracy_test > accuracy_best and epoch > 15:
            accuracy_best = accuracy_test
            save_path = os.path.join(args.model_root, args.name + '.pth')
            torch.save(model.state_dict(), save_path, )

        log_list.append(train_loss / len(train_loader))
        log_list.append(accuracy_test)
        log_list.append(accuracy_best)
        print(
            "Epoch {}, loss={:.5f}, accuracy_test={:.5f},  accuracy_best={:.5f}".format(epoch,
                                                                                        train_loss / len(train_loader),
                                                                                        accuracy_test, accuracy_best))
        train_loss = 0

        print(loss_kl_sum / len(train_loader))

        loss_kl_sum = 0

        if args.lr_decrease:
            if args.lr_warmup:
                scheduler_warmup.step(epoch=epoch)
            else:
                cos_scheduler.step(epoch=epoch)
        if epoch < 20:
            print(epoch, optimizer.param_groups[0]['lr'])

        # save model and para
        if epoch % save_interval == 0:
            train_state = {
                "Epoch": epoch,
                "model_state": model.state_dict(),
                "optim_state": optimizer.state_dict(),
                "args": args
            }
            models_dir = args.model_root + '/' + args.name + '.pt'
            torch.save(train_state, models_dir)

        #  save log
        with open(log_dir, 'a+', newline='') as f:
            # 训练结果
            my_writer = csv.writer(f)
            my_writer.writerow(log_list)
            log_list = []
        epoch = epoch + 1
    train_duration_sec = int(time.time() - start)
    print("training is end", train_duration_sec)


def get_std(std, index):
    index1 = index.view(index.shape[0], -1)
    index1 = torch.squeeze(index1, dim=1)
    std = std ** 2
    std = std.view(std.shape[0], -1)
    std = torch.mean(std, dim=1)
    # print(std.shape,index1.shape)
    std1 = torch.sum(std * index1) / torch.sum(index1)
    return std1


def train_base_multi_auxi_dul(model, cost, optimizer, train_loader, test_loader, args):
    '''
    适用于多模态分类的基础训练函数
    :param model:
    :param cost:
    :param optimizer:
    :param train_loader:
    :param test_loader:
    :param args:
    :return:
    '''
    print(args)

    # Initialize and open timer
    start = time.time()

    if not os.path.exists(args.model_root):
        os.makedirs(args.model_root)
    if not os.path.exists(args.log_root):
        os.makedirs(args.log_root)

    models_dir = args.model_root + '/' + args.name + '.pt'
    log_dir = args.log_root + '/' + args.name + '.csv'

    auxi_cross_entropy = nn.CrossEntropyLoss(reduction='none')

    # save args
    with open(log_dir, 'a+', newline='') as f:
        my_writer = csv.writer(f)
        args_dict = vars(args)
        for key, value in args_dict.items():
            my_writer.writerow([key, value])
        f.close()

    #  learning rate decay
    if args.lr_decrease == 'cos':
        print("lrcos is using")
        cos_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.train_epoch + 20, eta_min=0)

        if args.lr_warmup:
            scheduler_warmup = GradualWarmupScheduler(args, optimizer, multiplier=1,
                                                      after_scheduler=cos_scheduler)
    elif args.lr_decrease == 'multi_step':
        print("multi_step is using")
        cos_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[np.int(args.train_epoch * 1 / 6),
                                                                              np.int(args.train_epoch * 2 / 6),
                                                                              np.int(args.train_epoch * 3 / 6)])
        if args.lr_warmup:
            scheduler_warmup = GradualWarmupScheduler(args, optimizer, multiplier=1,
                                                      after_scheduler=cos_scheduler)
    else:
        if args.lr_warmup:
            scheduler_warmup = GradualWarmupScheduler(args, optimizer, multiplier=1)

    # Training initialization
    epoch_num = args.train_epoch
    log_interval = args.log_interval
    save_interval = args.save_interval
    batch_num = 0
    train_loss = 0
    epoch = 0
    accuracy_best = 0
    hter_best = 1
    acer_best = 1
    loss_kl_sum = 0
    log_list = []  # log need to save
    auxi_sum = 0
    fusion_sum = 0

    std1_sum = 0
    std2_sum = 0
    std3_sum = 0
    std4_sum = 0
    std5_sum = 0
    std6_sum = 0
    std7_sum = 0

    if args.retrain:
        if not os.path.exists(models_dir):
            print("no trained model")
        else:
            state_read = torch.load(models_dir)
            model.load_state_dict(state_read['model_state'])
            optimizer.load_state_dict(state_read['optim_state'])
            epoch = state_read['Epoch']
            print("retaining")

    # Train
    while epoch < epoch_num:
        for batch_idx, batch_sample in enumerate(
                tqdm(train_loader, desc="Epoch {}/{}".format(epoch, epoch_num))):

            batch_num += 1
            img_rgb, img_ir, img_depth, target = batch_sample['image_x'], batch_sample['image_ir'], \
                batch_sample['image_depth'], batch_sample[
                'binary_label']
            # if epoch == 0:
            #     continue
            if torch.cuda.is_available():
                img_rgb = img_rgb.cuda()
                img_ir = img_ir.cuda()
                img_depth = img_depth.cuda()
                target = target.cuda()

            # optimizer.zero_grad()
            for p in model.parameters():
                p.grad = None

            model.args.epoch = epoch
            output, layer3, layer4, x_rgb_out, x_ir_out, x_depth_out, p, (mul, std) = model(img_rgb, img_ir, img_depth)
            if isinstance(output, tuple):
                output = output[0]

            std1 = get_std(std, (p[:, 0]) * (1 - p[:, 1]) * (1 - p[:, 2]))
            std2 = get_std(std, (1 - p[:, 0]) * (p[:, 1]) * (1 - p[:, 2]))
            std3 = get_std(std, (1 - p[:, 0]) * (1 - p[:, 1]) * (p[:, 2]))
            std4 = get_std(std, (p[:, 0]) * (p[:, 1]) * (1 - p[:, 2]))
            std5 = get_std(std, (p[:, 0]) * (1 - p[:, 1]) * (p[:, 2]))
            std6 = get_std(std, (1 - p[:, 0]) * (p[:, 1]) * (p[:, 2]))
            std7 = get_std(std, (p[:, 0]) * (p[:, 1]) * (p[:, 2]))

            std1_sum += std1.item()
            std2_sum += std2.item()
            std3_sum += std3.item()
            std4_sum += std4.item()
            std5_sum += std5.item()
            std6_sum += std6.item()
            std7_sum += std7.item()

            if args.dataset == 'surf':

                fusion_loss = cost(output, target)

                x_rgb_loss_batch = auxi_cross_entropy(x_rgb_out, target)

                # x_auxi_weak = torch.sum(x_rgb_loss_batch * ((1 - p[:, 1]))) / p.shape[0]

                x_rgb_loss = torch.sum(x_rgb_loss_batch * ((p[:, 0]) * (1 - p[:, 1]) * (1 - p[:, 2]))) / p.shape[0]

                x_ir_loss_batch = auxi_cross_entropy(x_ir_out, target)

                # print(type(p[:, 0]))
                index = p[:, 0].int() | p[:, 2].int()
                index = index.float()
                # print(index)
                x_ir_loss = torch.sum(x_ir_loss_batch * ((p[:, 0]) * (1 - p[:, 1]) * (p[:, 2]))) / p.shape[0]

                x_depth_loss_batch = auxi_cross_entropy(x_depth_out, target)
                x_depth_loss = torch.sum(x_depth_loss_batch * ((p[:, 2]) * (1 - p[:, 0]) * (1 - p[:, 1]))) / p.shape[0]

            else:

                fusion_loss = cost(output, target)

                x_rgb_loss_batch = auxi_cross_entropy(x_rgb_out, target)

                # x_auxi_weak = torch.sum(x_rgb_loss_batch * ((1 - p[:, 0]))) / p.shape[0]
                x_rgb_loss = torch.sum(x_rgb_loss_batch * ((1 - p[:, 0]) * (1 - p[:, 1]) * (p[:, 2]))) / p.shape[0]

                x_ir_loss_batch = auxi_cross_entropy(x_ir_out, target)

                # print(type(p[:, 0]))
                index = p[:, 0].int() | p[:, 2].int()
                # index = index.float()
                # print(index)
                x_ir_loss = torch.sum(x_ir_loss_batch * ((1 - p[:, 2]) * (p[:, 1]) * (1 - p[:, 0]))) / p.shape[0]

                x_depth_loss_batch = auxi_cross_entropy(x_depth_out, target)
                x_depth_loss = torch.sum(x_depth_loss_batch * ((p[:, 1]) * (p[:, 2]) * (1 - p[:, 0]))) / p.shape[0]

            if epoch > 5:

                loss_cls = fusion_loss + args.auxi_scale * (x_rgb_loss + x_depth_loss + x_ir_loss)
            else:
                loss_cls = fusion_loss

            fusion_sum += fusion_loss.cpu().detach().numpy()
            auxi_sum += (x_rgb_loss + x_depth_loss + x_ir_loss).cpu().detach().numpy()

            variance_dul = std ** 2
            variance_dul = variance_dul.view(variance_dul.shape[0], -1)
            mul = mul.view(mul.shape[0], -1)

            loss_kl = torch.sum(((variance_dul + mul ** 2 - torch.log(variance_dul) - 1) * 0.5), dim=1)
            loss_kl = torch.mean(loss_kl)

            # if epoch > 5:
            #
            #     loss = loss_cls + args.kl_scale * loss_kl
            # else:
            #     loss = loss_cls

            loss = loss_cls + args.kl_scale * loss_kl

            train_loss += loss.item()
            loss_kl_sum += loss_kl.item()
            loss.backward()

            # if batch_num>10:
            #     print("weight.grad:", model.special_bone_rgb[0].weight.grad.mean(), model.special_bone_rgb[0].weight.grad.min(), model.special_bone_rgb[0].weight.grad.max())
            #     print("weight.grad:", model.special_bone_ir[0].weight.grad.mean(), model.special_bone_ir[0].weight.grad.min(), model.special_bone_ir[0].weight.grad.max())
            #     print("weight.grad:", model.special_bone_depth[0].weight.grad.mean(), model.special_bone_depth[0].weight.grad.min(), model.special_bone_depth[0].weight.grad.max())

            if (model.special_bone_rgb[0].weight.grad is None) or (model.special_bone_ir[0].weight.grad is None) or (
                    model.special_bone_depth[0].weight.grad is None):
                print("none!!!!!!none!!!!!")

            # print(model.special_bone_rgb[0].weight.grad)
            # print(model.special_bone_ir[0].weight.grad)
            # print(model.special_bone_depth[0].weight.grad)

            optimizer.step()

        # testing
        result_test, _ = calc_accuracy_multi(model, loader=test_loader, hter=True, verbose=True)
        hter_test = result_test[3]
        acer_test = result_test[-1]
        # save_path = os.path.join(args.model_root, args.name + "_epoch_" + str(epoch)
        #                          + '.pth')
        # torch.save(model.state_dict(), save_path)

        accuracy_test = result_test[0]

        if acer_test < acer_best and epoch > 30:
            acer_best = acer_test
            save_path = os.path.join(args.model_root, args.name + '_acer_best_' + '.pth')
            torch.save(model.state_dict(), save_path, )

        if hter_test < hter_best and epoch > 30:
            hter_best = hter_test
            save_path = os.path.join(args.model_root, args.name + '_hter_best_' + '.pth')
            torch.save(model.state_dict(), save_path, )

        if accuracy_test > accuracy_best and epoch > 30:
            accuracy_best = accuracy_test
            save_path = os.path.join(args.model_root, args.name + '.pth')
            torch.save(model.state_dict(), save_path, )
        log_list.append(train_loss / len(train_loader))
        log_list.append(accuracy_test)
        log_list.append(accuracy_best)
        print(
            "Epoch {}, loss={:.5f}, accuracy_test={:.5f},  accuracy_best={:.5f}".format(epoch,
                                                                                        train_loss / len(train_loader),
                                                                                        accuracy_test, accuracy_best))

        print(fusion_sum / len(train_loader), loss_kl_sum / len(train_loader), auxi_sum / (len(train_loader)))

        print(std1_sum/ len(train_loader), std2_sum/ len(train_loader), std3_sum/ len(train_loader), std4_sum/ len(train_loader), std5_sum/ len(train_loader), std6_sum/ len(train_loader), std7_sum/ len(train_loader))
        loss_kl_sum = 0
        train_loss = 0
        auxi_sum = 0
        fusion_sum = 0
        std1_sum = 0
        std2_sum = 0
        std3_sum = 0
        std4_sum = 0
        std5_sum = 0
        std6_sum = 0
        std7_sum = 0

        if args.lr_decrease:
            if args.lr_warmup:
                scheduler_warmup.step(epoch=epoch)
            else:
                cos_scheduler.step(epoch=epoch)
        if epoch < 20:
            print(epoch, optimizer.param_groups[0]['lr'])

        # save model and para
        if epoch % save_interval == 0:
            train_state = {
                "Epoch": epoch,
                "model_state": model.state_dict(),
                "optim_state": optimizer.state_dict(),
                "args": args
            }
            models_dir = args.model_root + '/' + args.name + '.pt'
            torch.save(train_state, models_dir)

        #  save log
        with open(log_dir, 'a+', newline='') as f:
            # 训练结果
            my_writer = csv.writer(f)
            my_writer.writerow(log_list)
            log_list = []
        epoch = epoch + 1
    train_duration_sec = int(time.time() - start)
    print("training is end", train_duration_sec)


def train_base_multi_auxi_dul_mix(model, cost, optimizer, train_loader, test_loader, args):
    '''
    适用于多模态分类的基础训练函数
    :param model:
    :param cost:
    :param optimizer:
    :param train_loader:
    :param test_loader:
    :param args:
    :return:
    '''
    print(args)

    # Initialize and open timer
    start = time.time()

    if not os.path.exists(args.model_root):
        os.makedirs(args.model_root)
    if not os.path.exists(args.log_root):
        os.makedirs(args.log_root)

    models_dir = args.model_root + '/' + args.name + '.pt'
    log_dir = args.log_root + '/' + args.name + '.csv'

    auxi_cross_entropy = nn.CrossEntropyLoss(reduction='none')

    # save args
    with open(log_dir, 'a+', newline='') as f:
        my_writer = csv.writer(f)
        args_dict = vars(args)
        for key, value in args_dict.items():
            my_writer.writerow([key, value])
        f.close()

    #  learning rate decay
    if args.lr_decrease == 'cos':
        print("lrcos is using")
        cos_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.train_epoch + 20, eta_min=0)

        if args.lr_warmup:
            scheduler_warmup = GradualWarmupScheduler(args, optimizer, multiplier=1,
                                                      after_scheduler=cos_scheduler)
    elif args.lr_decrease == 'multi_step':
        print("multi_step is using")
        cos_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[np.int(args.train_epoch * 1 / 6),
                                                                              np.int(args.train_epoch * 2 / 6),
                                                                              np.int(args.train_epoch * 3 / 6)])
        if args.lr_warmup:
            scheduler_warmup = GradualWarmupScheduler(args, optimizer, multiplier=1,
                                                      after_scheduler=cos_scheduler)
    else:
        if args.lr_warmup:
            scheduler_warmup = GradualWarmupScheduler(args, optimizer, multiplier=1)

    # Training initialization
    epoch_num = args.train_epoch
    log_interval = args.log_interval
    save_interval = args.save_interval
    batch_num = 0
    train_loss = 0
    epoch = 0
    accuracy_best = 0
    hter_best = 1
    acer_best = 1
    loss_kl_sum = 0
    log_list = []  # log need to save

    if args.retrain:
        if not os.path.exists(models_dir):
            print("no trained model")
        else:
            state_read = torch.load(models_dir)
            model.load_state_dict(state_read['model_state'])
            optimizer.load_state_dict(state_read['optim_state'])
            epoch = state_read['Epoch']
            print("retaining")

    # Train
    while epoch < epoch_num:
        for batch_idx, batch_sample in enumerate(
                tqdm(train_loader, desc="Epoch {}/{}".format(epoch, epoch_num))):

            batch_num += 1
            img_rgb, img_ir, img_depth, target = batch_sample['image_x'], batch_sample['image_ir'], \
                batch_sample['image_depth'], batch_sample[
                'binary_label']
            if epoch == 0:
                continue
            if torch.cuda.is_available():
                img_rgb = img_rgb.cuda()
                img_ir = img_ir.cuda()
                img_depth = img_depth.cuda()
                target = target.cuda()

            # optimizer.zero_grad()
            for p in model.parameters():
                p.grad = None

            model.args.epoch = epoch
            output, layer3, layer4, x_rgb_out, x_ir_out, x_depth_out, p, (mul, std) = model(img_rgb, img_ir, img_depth)
            if isinstance(output, tuple):
                output = output[0]

            mul = mul.view(mul.shape[0], -1)
            mul = torch.mean(mul, dim=1)
            std = std.view(std.shape[0], -1)
            std = torch.mean(std, dim=1)

            c = list(zip(output[:, 0], output[:, 1], target, mul, std))
            shuffle(c)
            shuffle_output1, shuffle_output_2, shuffle_target, shuffle_mul, shuffle_std = zip(*c)
            shuffle_output1 = torch.tensor(shuffle_output1).cuda().unsqueeze(dim=1)
            shuffle_output_2 = torch.tensor(shuffle_output_2).cuda().unsqueeze(dim=1)
            # print(shuffle_output1)

            shuffle_output = torch.cat((shuffle_output1, shuffle_output_2), dim=1)

            # print(shuffle_output)
            shuffle_target = torch.tensor(shuffle_target).cuda()
            shuffle_mul = torch.tensor(shuffle_mul).cuda()
            shuffle_std = torch.tensor(shuffle_std).cuda()
            # shuffle_output=[torch.tensor(data) for data in shuffle_output]

            std1 = std / (shuffle_std + std)
            std2 = shuffle_std / (std + shuffle_std)
            std1 = std1.unsqueeze(dim=1)
            std2 = std2.unsqueeze(dim=1)
            # print(std1.shape)
            # print(std2.shape)
            # print(std1)
            # print(std2)

            output_mix = std1 * output + std2 * shuffle_output

            loss_cls_1 = cost(output_mix, target)

            loss_cls_2 = cost(output_mix, shuffle_target)

            if args.dataset == 'surf':

                fusion_loss = loss_cls_1 + loss_cls_2

                x_rgb_loss_batch = auxi_cross_entropy(x_rgb_out, target)

                # x_auxi_weak = torch.sum(x_rgb_loss_batch * ((1 - p[:, 1]))) / p.shape[0]

                x_rgb_loss = torch.sum(x_rgb_loss_batch * ((p[:, 0]) * (1 - p[:, 1]) * (1 - p[:, 2]))) / p.shape[0]

                x_ir_loss_batch = auxi_cross_entropy(x_ir_out, target)

                # print(type(p[:, 0]))
                index = p[:, 0].int() | p[:, 2].int()
                index = index.float()
                # print(index)
                x_ir_loss = torch.sum(x_ir_loss_batch * ((p[:, 0]) * (1 - p[:, 1]) * (p[:, 2]))) / p.shape[0]

                x_depth_loss_batch = auxi_cross_entropy(x_depth_out, target)
                x_depth_loss = torch.sum(x_depth_loss_batch * ((p[:, 2]) * (1 - p[:, 0]) * (1 - p[:, 1]))) / p.shape[0]

            else:

                fusion_loss = loss_cls_1 + loss_cls_2

                x_rgb_loss_batch = auxi_cross_entropy(x_rgb_out, target)

                # x_auxi_weak = torch.sum(x_rgb_loss_batch * ((1 - p[:, 0]))) / p.shape[0]
                x_rgb_loss = torch.sum(x_rgb_loss_batch * ((1 - p[:, 0]) * (1 - p[:, 1]) * (p[:, 2]))) / p.shape[0]

                x_ir_loss_batch = auxi_cross_entropy(x_ir_out, target)

                # print(type(p[:, 0]))
                index = p[:, 0].int() | p[:, 2].int()
                # index = index.float()
                # print(index)
                x_ir_loss = torch.sum(x_ir_loss_batch * ((1 - p[:, 2]) * (p[:, 1]) * (1 - p[:, 0]))) / p.shape[0]

                x_depth_loss_batch = auxi_cross_entropy(x_depth_out, target)
                x_depth_loss = torch.sum(x_depth_loss_batch * ((p[:, 1]) * (p[:, 2]) * (1 - p[:, 0]))) / p.shape[0]

            loss_cls = fusion_loss + x_rgb_loss + x_depth_loss + x_ir_loss

            # variance_dul = std ** 2
            # variance_dul = variance_dul.view(variance_dul.shape[0], -1)
            # mul = mul.view(mul.shape[0], -1)
            #
            # loss_kl = torch.sum(((variance_dul + mul ** 2 - torch.log(variance_dul) - 1) * 0.5), dim=1)
            # loss_kl = torch.mean(loss_kl)

            loss = loss_cls

            train_loss += loss.item()
            # loss_kl_sum += loss_kl.item()
            loss.backward()

            # if batch_num>10:
            #     print("weight.grad:", model.special_bone_rgb[0].weight.grad.mean(), model.special_bone_rgb[0].weight.grad.min(), model.special_bone_rgb[0].weight.grad.max())
            #     print("weight.grad:", model.special_bone_ir[0].weight.grad.mean(), model.special_bone_ir[0].weight.grad.min(), model.special_bone_ir[0].weight.grad.max())
            #     print("weight.grad:", model.special_bone_depth[0].weight.grad.mean(), model.special_bone_depth[0].weight.grad.min(), model.special_bone_depth[0].weight.grad.max())

            if (model.special_bone_rgb[0].weight.grad is None) or (model.special_bone_ir[0].weight.grad is None) or (
                    model.special_bone_depth[0].weight.grad is None):
                print("none!!!!!!none!!!!!")

            # print(model.special_bone_rgb[0].weight.grad)
            # print(model.special_bone_ir[0].weight.grad)
            # print(model.special_bone_depth[0].weight.grad)

            optimizer.step()

        # testing
        result_test, _ = calc_accuracy_multi(model, loader=test_loader, hter=True, verbose=True)
        hter_test = result_test[3]
        acer_test = result_test[-1]
        # save_path = os.path.join(args.model_root, args.name + "_epoch_" + str(epoch)
        #                          + '.pth')
        # torch.save(model.state_dict(), save_path)

        accuracy_test = result_test[0]

        if acer_test < acer_best and epoch > 15:
            acer_best = acer_test
            save_path = os.path.join(args.model_root, args.name + '_acer_best_' + '.pth')
            torch.save(model.state_dict(), save_path, )

        if hter_test < hter_best and epoch > 15:
            hter_best = hter_test
            save_path = os.path.join(args.model_root, args.name + '_hter_best_' + '.pth')
            torch.save(model.state_dict(), save_path, )

        if accuracy_test > accuracy_best and epoch > 15:
            accuracy_best = accuracy_test
            save_path = os.path.join(args.model_root, args.name + '.pth')
            torch.save(model.state_dict(), save_path, )
        log_list.append(train_loss / len(train_loader))
        log_list.append(accuracy_test)
        log_list.append(accuracy_best)
        print(
            "Epoch {}, loss={:.5f}, accuracy_test={:.5f},  accuracy_best={:.5f}".format(epoch,
                                                                                        train_loss / len(train_loader),
                                                                                        accuracy_test, accuracy_best))

        print(loss_kl_sum / len(train_loader))

        loss_kl_sum = 0
        train_loss = 0

        if args.lr_decrease:
            if args.lr_warmup:
                scheduler_warmup.step(epoch=epoch)
            else:
                cos_scheduler.step(epoch=epoch)
        if epoch < 20:
            print(epoch, optimizer.param_groups[0]['lr'])

        # save model and para
        if epoch % save_interval == 0:
            train_state = {
                "Epoch": epoch,
                "model_state": model.state_dict(),
                "optim_state": optimizer.state_dict(),
                "args": args
            }
            models_dir = args.model_root + '/' + args.name + '.pt'
            torch.save(train_state, models_dir)

        #  save log
        with open(log_dir, 'a+', newline='') as f:
            # 训练结果
            my_writer = csv.writer(f)
            my_writer.writerow(log_list)
            log_list = []
        epoch = epoch + 1
    train_duration_sec = int(time.time() - start)
    print("training is end", train_duration_sec)


def train_base_multi_auxi_dul_uncertainty(model, cost, optimizer, train_loader, test_loader, args):
    '''
    适用于多模态分类的基础训练函数
    :param model:
    :param cost:
    :param optimizer:
    :param train_loader:
    :param test_loader:
    :param args:
    :return:
    '''
    print(args)

    # Initialize and open timer
    start = time.time()

    if not os.path.exists(args.model_root):
        os.makedirs(args.model_root)
    if not os.path.exists(args.log_root):
        os.makedirs(args.log_root)

    models_dir = args.model_root + '/' + args.name + '.pt'
    log_dir = args.log_root + '/' + args.name + '.csv'

    auxi_cross_entropy = nn.CrossEntropyLoss(reduction='none')

    # save args
    with open(log_dir, 'a+', newline='') as f:
        my_writer = csv.writer(f)
        args_dict = vars(args)
        for key, value in args_dict.items():
            my_writer.writerow([key, value])
        f.close()

    #  learning rate decay
    if args.lr_decrease == 'cos':
        print("lrcos is using")
        cos_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.train_epoch + 20, eta_min=0)

        if args.lr_warmup:
            scheduler_warmup = GradualWarmupScheduler(args, optimizer, multiplier=1,
                                                      after_scheduler=cos_scheduler)
    elif args.lr_decrease == 'multi_step':
        print("multi_step is using")
        cos_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[np.int(args.train_epoch * 1 / 6),
                                                                              np.int(args.train_epoch * 2 / 6),
                                                                              np.int(args.train_epoch * 3 / 6)])
        if args.lr_warmup:
            scheduler_warmup = GradualWarmupScheduler(args, optimizer, multiplier=1,
                                                      after_scheduler=cos_scheduler)
    else:
        if args.lr_warmup:
            scheduler_warmup = GradualWarmupScheduler(args, optimizer, multiplier=1)

    # Training initialization
    epoch_num = args.train_epoch
    log_interval = args.log_interval
    save_interval = args.save_interval
    batch_num = 0
    train_loss = 0
    epoch = 0
    accuracy_best = 0
    hter_best = 1
    acer_best = 1
    loss_kl_sum = 0
    log_list = []  # log need to save

    if args.retrain:
        if not os.path.exists(models_dir):
            print("no trained model")
        else:
            state_read = torch.load(models_dir)
            model.load_state_dict(state_read['model_state'])
            optimizer.load_state_dict(state_read['optim_state'])
            epoch = state_read['Epoch']
            print("retaining")

    # Train
    while epoch < epoch_num:
        for batch_idx, batch_sample in enumerate(
                tqdm(train_loader, desc="Epoch {}/{}".format(epoch, epoch_num))):

            batch_num += 1
            img_rgb, img_ir, img_depth, target = batch_sample['image_x'], batch_sample['image_ir'], \
                batch_sample['image_depth'], batch_sample[
                'binary_label']
            if epoch == 0:
                continue
            if torch.cuda.is_available():
                img_rgb = img_rgb.cuda()
                img_ir = img_ir.cuda()
                img_depth = img_depth.cuda()
                target = target.cuda()

            # optimizer.zero_grad()
            for p in model.parameters():
                p.grad = None

            model.args.epoch = epoch
            output, layer3, layer4, x_rgb_out, x_ir_out, x_depth_out, p, (mul, std) = model(img_rgb, img_ir, img_depth)
            if isinstance(output, tuple):
                output = output[0]

            if args.dataset == 'surf':

                fusion_loss = cost(output, target)

                x_rgb_loss_batch = auxi_cross_entropy(x_rgb_out, target)

                # x_auxi_weak = torch.sum(x_rgb_loss_batch * ((1 - p[:, 1]))) / p.shape[0]

                x_rgb_loss = torch.sum(x_rgb_loss_batch * ((p[:, 0]) * (1 - p[:, 1]) * (1 - p[:, 2]))) / p.shape[0]

                x_ir_loss_batch = auxi_cross_entropy(x_ir_out, target)

                # print(type(p[:, 0]))
                index = p[:, 0].int() | p[:, 2].int()
                index = index.float()
                # print(index)
                x_ir_loss = torch.sum(x_ir_loss_batch * ((p[:, 0]) * (1 - p[:, 1]) * (p[:, 2]))) / p.shape[0]

                x_depth_loss_batch = auxi_cross_entropy(x_depth_out, target)
                x_depth_loss = torch.sum(x_depth_loss_batch * ((p[:, 2]) * (1 - p[:, 0]) * (1 - p[:, 1]))) / p.shape[0]

            else:

                fusion_loss = cost(output, target)

                x_rgb_loss_batch = auxi_cross_entropy(x_rgb_out, target)

                # x_auxi_weak = torch.sum(x_rgb_loss_batch * ((1 - p[:, 0]))) / p.shape[0]
                x_rgb_loss = torch.sum(x_rgb_loss_batch * ((1 - p[:, 0]) * (1 - p[:, 1]) * (p[:, 2]))) / p.shape[0]

                x_ir_loss_batch = auxi_cross_entropy(x_ir_out, target)

                # print(type(p[:, 0]))
                index = p[:, 0].int() | p[:, 2].int()
                # index = index.float()
                # print(index)
                x_ir_loss = torch.sum(x_ir_loss_batch * (1 - (p[:, 2]) * (p[:, 1]) * (1 - p[:, 0]))) / p.shape[0]

                x_depth_loss_batch = auxi_cross_entropy(x_depth_out, target)
                x_depth_loss = torch.sum(x_depth_loss_batch * ((p[:, 1]) * (p[:, 2]) * (1 - p[:, 0]))) / p.shape[0]

            # loss_cls = fusion_loss + x_rgb_loss + x_depth_loss + x_ir_loss

            variance_dul = std ** 2
            variance_dul = variance_dul.view(variance_dul.shape[0], -1)
            mul = mul.view(mul.shape[0], -1)

            loss_kl = torch.sum(((variance_dul + mul ** 2 - torch.log(variance_dul) - 1) * 0.5), dim=1)
            loss_kl = torch.mean(loss_kl)

            std = std.view(std.shape[0], -1)
            std = torch.mean(std, dim=1)
            uncertainty_weight = std / torch.sum(std)
            # print(uncertainty_weight)

            loss_cls = torch.sum(fusion_loss * uncertainty_weight)
            loss = loss_cls + args.kl_scale * loss_kl

            train_loss += loss.item()
            loss_kl_sum += loss_kl.item()
            loss.backward()

            # if batch_num>10:
            #     print("weight.grad:", model.special_bone_rgb[0].weight.grad.mean(), model.special_bone_rgb[0].weight.grad.min(), model.special_bone_rgb[0].weight.grad.max())
            #     print("weight.grad:", model.special_bone_ir[0].weight.grad.mean(), model.special_bone_ir[0].weight.grad.min(), model.special_bone_ir[0].weight.grad.max())
            #     print("weight.grad:", model.special_bone_depth[0].weight.grad.mean(), model.special_bone_depth[0].weight.grad.min(), model.special_bone_depth[0].weight.grad.max())

            if (model.special_bone_rgb[0].weight.grad is None) or (model.special_bone_ir[0].weight.grad is None) or (
                    model.special_bone_depth[0].weight.grad is None):
                print("none!!!!!!none!!!!!")

            # print(model.special_bone_rgb[0].weight.grad)
            # print(model.special_bone_ir[0].weight.grad)
            # print(model.special_bone_depth[0].weight.grad)

            optimizer.step()

        # testing
        result_test, _ = calc_accuracy_multi(model, loader=test_loader, hter=True, verbose=True)
        hter_test = result_test[3]
        acer_test = result_test[-1]
        # save_path = os.path.join(args.model_root, args.name + "_epoch_" + str(epoch)
        #                          + '.pth')
        # torch.save(model.state_dict(), save_path)

        accuracy_test = result_test[0]

        if acer_test < acer_best and epoch > 15:
            acer_best = acer_test
            save_path = os.path.join(args.model_root, args.name + '_acer_best_' + '.pth')
            torch.save(model.state_dict(), save_path, )

        if hter_test < hter_best and epoch > 15:
            hter_best = hter_test
            save_path = os.path.join(args.model_root, args.name + '_hter_best_' + '.pth')
            torch.save(model.state_dict(), save_path, )

        if accuracy_test > accuracy_best and epoch > 15:
            accuracy_best = accuracy_test
            save_path = os.path.join(args.model_root, args.name + '.pth')
            torch.save(model.state_dict(), save_path, )
        log_list.append(train_loss / len(train_loader))
        log_list.append(accuracy_test)
        log_list.append(accuracy_best)
        print(
            "Epoch {}, loss={:.5f}, accuracy_test={:.5f},  accuracy_best={:.5f}".format(epoch,
                                                                                        train_loss / len(train_loader),
                                                                                        accuracy_test, accuracy_best))

        print(loss_kl_sum / len(train_loader))

        loss_kl_sum = 0
        train_loss = 0

        if args.lr_decrease:
            if args.lr_warmup:
                scheduler_warmup.step(epoch=epoch)
            else:
                cos_scheduler.step(epoch=epoch)
        if epoch < 20:
            print(epoch, optimizer.param_groups[0]['lr'])

        # save model and para
        if epoch % save_interval == 0:
            train_state = {
                "Epoch": epoch,
                "model_state": model.state_dict(),
                "optim_state": optimizer.state_dict(),
                "args": args
            }
            models_dir = args.model_root + '/' + args.name + '.pt'
            torch.save(train_state, models_dir)

        #  save log
        with open(log_dir, 'a+', newline='') as f:
            # 训练结果
            my_writer = csv.writer(f)
            my_writer.writerow(log_list)
            log_list = []
        epoch = epoch + 1
    train_duration_sec = int(time.time() - start)
    print("training is end", train_duration_sec)


def train_base_multi_auxi(model, cost, optimizer, train_loader, test_loader, args):
    '''
    适用于多模态分类的基础训练函数
    :param model:
    :param cost:
    :param optimizer:
    :param train_loader:
    :param test_loader:
    :param args:
    :return:
    '''
    print(args)

    # Initialize and open timer
    start = time.time()

    if not os.path.exists(args.model_root):
        os.makedirs(args.model_root)
    if not os.path.exists(args.log_root):
        os.makedirs(args.log_root)

    models_dir = args.model_root + '/' + args.name + '.pt'
    log_dir = args.log_root + '/' + args.name + '.csv'

    auxi_cross_entropy = nn.CrossEntropyLoss(reduction='none')

    # save args
    with open(log_dir, 'a+', newline='') as f:
        my_writer = csv.writer(f)
        args_dict = vars(args)
        for key, value in args_dict.items():
            my_writer.writerow([key, value])
        f.close()

    #  learning rate decay
    if args.lr_decrease == 'cos':
        print("lrcos is using")
        cos_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.train_epoch + 20, eta_min=0)

        if args.lr_warmup:
            scheduler_warmup = GradualWarmupScheduler(args, optimizer, multiplier=1,
                                                      after_scheduler=cos_scheduler)
    elif args.lr_decrease == 'multi_step':
        print("multi_step is using")
        cos_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[int(args.train_epoch * 1 / 6),
                                                                              int(args.train_epoch * 2 / 6),
                                                                              int(args.train_epoch * 3 / 6)])
        if args.lr_warmup:
            scheduler_warmup = GradualWarmupScheduler(args, optimizer, multiplier=1,
                                                      after_scheduler=cos_scheduler)
    else:
        if args.lr_warmup:
            scheduler_warmup = GradualWarmupScheduler(args, optimizer, multiplier=1)

    # Training initialization
    epoch_num = args.train_epoch
    log_interval = args.log_interval
    save_interval = args.save_interval
    batch_num = 0
    train_loss = 0
    epoch = 0
    accuracy_best = 0
    log_list = []  # log need to save

    if args.retrain:
        if not os.path.exists(models_dir):
            print("no trained model")
        else:
            state_read = torch.load(models_dir)
            model.load_state_dict(state_read['model_state'])
            optimizer.load_state_dict(state_read['optim_state'])
            epoch = state_read['Epoch']
            print("retaining")

    # Train
    while epoch < epoch_num:
        for batch_idx, batch_sample in enumerate(
                tqdm(train_loader, desc="Epoch {}/{}".format(epoch, epoch_num))):

            batch_num += 1
            img_rgb, img_ir, img_depth, target = batch_sample['image_x'], batch_sample['image_ir'], \
                batch_sample['image_depth'], batch_sample[
                'binary_label']
            # if epoch == 0:
            #     continue
            if torch.cuda.is_available():
                img_rgb = img_rgb.cuda()
                img_ir = img_ir.cuda()
                img_depth = img_depth.cuda()
                target = target.cuda()

            # optimizer.zero_grad()
            for p in model.parameters():
                p.grad = None

            model.args.epoch = epoch
            output, layer3, layer4, x_rgb_out, x_ir_out, x_depth_out, p = model(img_rgb, img_ir, img_depth)
            if isinstance(output, tuple):
                output = output[0]

            fusion_loss = cost(output, target)

            x_rgb_loss_batch = auxi_cross_entropy(x_rgb_out, target)

            # print(x_rgb_loss_batch.shape,p.shape)
            # print(p)

            x_rgb_loss = torch.sum(x_rgb_loss_batch * (p[:, 0] * (1 - p[:, 1]) * (1 - p[:, 2]))) / p.shape[0]

            x_ir_loss_batch = auxi_cross_entropy(x_ir_out, target)
            x_ir_loss = torch.sum(x_ir_loss_batch * ((1 - p[:, 0]) * (p[:, 1]) * (1 - p[:, 2]))) / p.shape[0]

            x_depth_loss_batch = auxi_cross_entropy(x_depth_out, target)
            x_depth_loss = torch.sum(x_depth_loss_batch * ((1 - p[:, 0]) * (1 - p[:, 1]) * (p[:, 2]))) / p.shape[0]

            loss = fusion_loss + x_rgb_loss + x_depth_loss + x_ir_loss

            train_loss += loss.item()
            loss.backward()

            # if batch_num>10:
            #     print("weight.grad:", model.special_bone_rgb[0].weight.grad.mean(), model.special_bone_rgb[0].weight.grad.min(), model.special_bone_rgb[0].weight.grad.max())
            #     print("weight.grad:", model.special_bone_ir[0].weight.grad.mean(), model.special_bone_ir[0].weight.grad.min(), model.special_bone_ir[0].weight.grad.max())
            #     print("weight.grad:", model.special_bone_depth[0].weight.grad.mean(), model.special_bone_depth[0].weight.grad.min(), model.special_bone_depth[0].weight.grad.max())

            if (model.special_bone_rgb[0].weight.grad is None) or (model.special_bone_ir[0].weight.grad is None) or (
                    model.special_bone_depth[0].weight.grad is None):
                print("none!!!!!!none!!!!!")

            # print(model.special_bone_rgb[0].weight.grad)
            # print(model.special_bone_ir[0].weight.grad)
            # print(model.special_bone_depth[0].weight.grad)

            optimizer.step()

        # testing
        result_test, _ = calc_accuracy_multi(model, loader=test_loader, hter=True, verbose=True)

        save_path = os.path.join(args.model_root, args.name + "_epoch_" + str(epoch)
                                 + '.pth')
        torch.save(model.state_dict(), save_path)

        accuracy_test = result_test[0]
        if accuracy_test > accuracy_best and epoch > 5:
            accuracy_best = accuracy_test
            save_path = os.path.join(args.model_root, args.name + '.pth')
            torch.save(model.state_dict(), save_path)
        log_list.append(train_loss / len(train_loader))
        log_list.append(accuracy_test)
        log_list.append(accuracy_best)
        print(
            "Epoch {}, loss={:.5f}, accuracy_test={:.5f},  accuracy_best={:.5f}".format(epoch,
                                                                                        train_loss / len(train_loader),
                                                                                        accuracy_test, accuracy_best))
        train_loss = 0

        if args.lr_decrease:
            if args.lr_warmup:
                scheduler_warmup.step(epoch=epoch)
            else:
                cos_scheduler.step(epoch=epoch)
        if epoch < 20:
            print(epoch, optimizer.param_groups[0]['lr'])

        # save model and para
        if epoch % save_interval == 0:
            train_state = {
                "Epoch": epoch,
                "model_state": model.state_dict(),
                "optim_state": optimizer.state_dict(),
                "args": args
            }
            models_dir = args.model_root + '/' + args.name + '.pt'
            torch.save(train_state, models_dir)

        #  save log
        with open(log_dir, 'a+', newline='') as f:
            # 训练结果
            my_writer = csv.writer(f)
            my_writer.writerow(log_list)
            log_list = []
        epoch = epoch + 1
    train_duration_sec = int(time.time() - start)
    print("training is end", train_duration_sec)


def train_ensemble(model, cost, optimizer, train_loader, test_loader, args):
    '''
    适用于多模态分类的基础训练函数
    :param model:
    :param cost:
    :param optimizer:
    :param train_loader:
    :param test_loader:
    :param args:
    :return:
    '''
    print(args)

    # Initialize and open timer
    start = time.time()

    if not os.path.exists(args.model_root):
        os.makedirs(args.model_root)
    if not os.path.exists(args.log_root):
        os.makedirs(args.log_root)

    models_dir = args.model_root + '/' + args.name + '.pt'
    log_dir = args.log_root + '/' + args.name + '.csv'

    # save args
    with open(log_dir, 'a+', newline='') as f:
        my_writer = csv.writer(f)
        args_dict = vars(args)
        for key, value in args_dict.items():
            my_writer.writerow([key, value])
        f.close()

    #  learning rate decay
    if args.lr_decrease == 'cos':
        print("lrcos is using")
        cos_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.train_epoch + 20, eta_min=0)

        if args.lr_warmup:
            scheduler_warmup = GradualWarmupScheduler(args, optimizer, multiplier=1,
                                                      after_scheduler=cos_scheduler)
    elif args.lr_decrease == 'multi_step':
        print("multi_step is using")
        cos_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[np.int(args.train_epoch * 1 / 6),
                                                                              np.int(args.train_epoch * 2 / 6),
                                                                              np.int(args.train_epoch * 3 / 6)])
        if args.lr_warmup:
            scheduler_warmup = GradualWarmupScheduler(args, optimizer, multiplier=1,
                                                      after_scheduler=cos_scheduler)

    # Training initialization
    epoch_num = args.train_epoch
    log_interval = args.log_interval
    save_interval = args.save_interval
    batch_num = 0
    train_loss = 0
    epoch = 0
    accuracy_best = 0
    acer_best = 1
    hter_best = 1
    log_list = []  # log need to save

    if args.retrain:
        if not os.path.exists(models_dir):
            print("no trained model")
        else:
            state_read = torch.load(models_dir)
            model.load_state_dict(state_read['model_state'])
            optimizer.load_state_dict(state_read['optim_state'])
            epoch = state_read['Epoch']
            print("retaining")

    # Train
    while epoch < epoch_num:
        for batch_idx, batch_sample in enumerate(
                tqdm(train_loader, desc="Epoch {}/{}".format(epoch, epoch_num))):

            batch_num += 1
            img_rgb, img_ir, img_depth, target = batch_sample['image_x'], batch_sample['image_ir'], \
                batch_sample['image_depth'], batch_sample[
                'binary_label']
            if epoch == 0:
                continue
            if torch.cuda.is_available():
                img_rgb = img_rgb.cuda()
                img_ir = img_ir.cuda()
                img_depth = img_depth.cuda()
                target = target.cuda()

            optimizer.zero_grad()

            output = model(img_rgb)

            loss = cost(output, target)

            train_loss += loss.item()
            loss.backward()
            optimizer.step()

        # testing
        print(model.fuse_weight_1.cpu())
        result_test = calc_accuracy_ensemble(model, loader=test_loader, hter=True)
        print(result_test)
        accuracy_test = result_test[0]
        hter_test = result_test[3]
        acer_test = result_test[-1]

        if acer_test < acer_best and epoch > 0:
            acer_best = acer_test
            save_path = os.path.join(args.model_root, args.name + '_acer_best_' + '.pth')
            torch.save(model.state_dict(), save_path, )

        if hter_test < hter_best and epoch > 0:
            hter_best = hter_test
            save_path = os.path.join(args.model_root, args.name + '_hter_best_' + '.pth')
            torch.save(model.state_dict(), save_path, )

        if acer_test < hter_test and acer_test < 0.65 and epoch % 2 == 0:
            save_path = os.path.join(args.model_root, args.name + '_' + str(epoch) + '_.pth')
            torch.save(model.state_dict(), save_path, )

        if accuracy_test > accuracy_best and epoch > 0:
            accuracy_best = accuracy_test
            save_path = os.path.join(args.model_root, args.name + '.pth')
            torch.save(model.state_dict(), save_path, )
        log_list.append(train_loss / len(train_loader))
        log_list.append(accuracy_test)
        log_list.append(accuracy_best)
        print(
            "Epoch {}, loss={:.5f}, accuracy_test={:.5f},  accuracy_best={:.5f}".format(epoch,
                                                                                        train_loss / len(train_loader),
                                                                                        accuracy_test, accuracy_best))
        train_loss = 0

        if args.lr_decrease:
            if args.lr_warmup:
                scheduler_warmup.step(epoch=epoch)
            else:
                cos_scheduler.step(epoch=epoch)
        if epoch < 20:
            print(epoch, optimizer.param_groups[0]['lr'])

        # save model and para
        if epoch % save_interval == 0:
            train_state = {
                "Epoch": epoch,
                "model_state": model.state_dict(),
                "optim_state": optimizer.state_dict(),
                "args": args
            }
            models_dir = args.model_root + '/' + args.name + '.pt'
            torch.save(train_state, models_dir)

        #  save log
        with open(log_dir, 'a+', newline='') as f:
            # 训练结果
            my_writer = csv.writer(f)
            my_writer.writerow(log_list)
            log_list = []
        epoch = epoch + 1
    train_duration_sec = int(time.time() - start)
    print("training is end", train_duration_sec)


def train_pixel_supervise(model, cost, optimizer, train_loader, test_loader, args):
    '''

    :param model:
    :param cost:
    :param optimizer:
    :param train_loader:
    :param test_loader:
    :param args:
    :return:
    '''

    print(args)

    # Initialize and open timer
    start = time.time()

    if not os.path.exists(args.model_root):
        os.makedirs(args.model_root)
    if not os.path.exists(args.log_root):
        os.makedirs(args.log_root)

    models_dir = args.model_root + '/' + args.name + '.pt'
    log_dir = args.log_root + '/' + args.name + '.csv'

    # save args
    with open(log_dir, 'a+', newline='') as f:
        my_writer = csv.writer(f)
        args_dict = vars(args)
        for key, value in args_dict.items():
            my_writer.writerow([key, value])
        f.close()

    # Cosine learning rate decay
    if args.lr_decrease:
        print("lrcos is using")
        cos_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.train_epoch, eta_min=0)

        if args.lr_warmup:
            scheduler_warmup = GradualWarmupScheduler(args, optimizer, multiplier=1,
                                                      after_scheduler=cos_scheduler)

    # Training initialization
    epoch_num = args.train_epoch
    log_interval = args.log_interval
    save_interval = args.save_interval
    batch_num = 0
    train_loss = 0
    epoch = 0
    loss_best = 1e4
    log_list = []  # log need to save

    if args.retrain:
        if not os.path.exists(models_dir):
            print("no trained model")
        else:
            state_read = torch.load(models_dir)
            model.load_state_dict(state_read['model_state'])
            optimizer.load_state_dict(state_read['optim_state'])
            epoch = state_read['Epoch']
            print("retaining")

    # Train
    while epoch < epoch_num:
        for batch_idx, (data, target) in enumerate(
                tqdm(train_loader, desc="Epoch {}/{}".format(epoch, epoch_num))):

            batch_num += 1

            target = torch.unsqueeze(target, dim=1)

            if torch.cuda.is_available():
                data, target = data.cuda(), target.cuda()

            if args.mixup:
                mixup_alpha = args.mixup_alpha
                inputs, labels_a, labels_b, lam = mixup_data(data, target, alpha=mixup_alpha)

            optimizer.zero_grad()

            output = model(data)

            if args.mixup:
                loss = mixup_criterion(cost, output, labels_a, labels_b, lam)
            else:
                loss = cost(output, target)

            train_loss += loss.item()
            loss.backward()
            optimizer.step()
            # if batch_idx % log_interval == 0:  # 准备打印相关信息，args.log_interval是最开头设置的好了的参数
            #     print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            #         epoch, batch_idx * len(data), len(train_loader.dataset),
            #                100. * batch_idx / len(train_loader), loss.item()))

        # testing
        result_test = calc_accuracy_pixel(model, loader=test_loader)
        test_loss = result_test
        if test_loss < loss_best:
            loss_best = train_loss / len(train_loader)
            save_path = args.model_root + args.name + '.pth'
            torch.save(model.state_dict(), save_path)
        log_list.append(test_loss)

        print(
            "Epoch {}, loss={:.5f}".format(epoch,
                                           train_loss / len(train_loader),
                                           ))
        train_loss = 0
        if args.lr_decrease:
            if args.lr_warmup:
                scheduler_warmup.step(epoch=epoch)
            else:
                cos_scheduler.step(epoch=epoch)
        if epoch < 20:
            print(epoch, optimizer.param_groups[0]['lr'])

        # save model and para
        if epoch % save_interval == 0:
            train_state = {
                "Epoch": epoch,
                "model_state": model.state_dict(),
                "optim_state": optimizer.state_dict(),
                "args": args
            }
            models_dir = args.model_root + '/' + args.name + '.pt'
            torch.save(train_state, models_dir)

        #  save log
        with open(log_dir, 'a+', newline='') as f:
            # 训练结果
            my_writer = csv.writer(f)
            my_writer.writerow(log_list)
            log_list = []
        epoch = epoch + 1
    train_duration_sec = int(time.time() - start)
    print("training is end", train_duration_sec)


def train_knowledge_distill_patch(net_dict, cost_dict, optimizer, train_loader, test_loader, args):
    '''

    :param model:
    :param cost:
    :param optimizer:
    :param train_loader:
    :param test_loader:
    :param args:
    :return:
    '''

    print(args)
    if not os.path.exists(args.model_root):
        os.makedirs(args.model_root)
    if not os.path.exists(args.log_root):
        os.makedirs(args.log_root)

    models_dir = args.model_root + '/' + args.name + '.pt'
    log_dir = args.log_root + '/' + args.name + '.csv'

    # save args
    with open(log_dir, 'a+', newline='') as f:
        my_writer = csv.writer(f)
        args_dict = vars(args)
        for key, value in args_dict.items():
            my_writer.writerow([key, value])
        f.close()

    student_model = net_dict['snet']
    teacher_model = net_dict['tnet']

    criterionCls = cost_dict['criterionCls']
    criterionKD = cost_dict['criterionKD']

    from loss.kd.pkt import PKTCosSim

    if torch.cuda.is_available():
        mmd_loss = MMD_loss().cuda()
    else:
        mmd_loss = MMD_loss()
    if torch.cuda.is_available():
        bce_loss = nn.BCELoss().cuda()
    else:
        bce_loss = nn.BCELoss()

    if torch.cuda.is_available():
        pkd_loss = PKTCosSim().cuda()
    else:
        pkd_loss = PKTCosSim()

    #  learning rate decay
    if args.lr_decrease == 'cos':
        print("lrcos is using")
        cos_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.train_epoch + 20, eta_min=0)

        if args.lr_warmup:
            scheduler_warmup = GradualWarmupScheduler(args, optimizer, multiplier=1,
                                                      after_scheduler=cos_scheduler)
    elif args.lr_decrease == 'multi_step':
        print("multi_step is using")
        cos_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[np.int(args.train_epoch * 1 / 6),
                                                                              np.int(args.train_epoch * 2 / 6),
                                                                              np.int(args.train_epoch * 3 / 6)])
        if args.lr_warmup:
            scheduler_warmup = GradualWarmupScheduler(args, optimizer, multiplier=1,
                                                      after_scheduler=cos_scheduler)

    # Training initialization
    epoch_num = args.train_epoch
    log_interval = args.log_interval
    save_interval = args.save_interval
    batch_num = 0
    train_loss = 0
    cls_loss_sum = 0
    kd_loss_sum = 0
    epoch = 0
    accuracy_best = 0
    acer_best = 1
    hter_best = 1
    log_list = []  # log need to save

    if args.retrain:
        if not os.path.exists(models_dir):
            print("no trained model")
        else:
            state_read = torch.load(models_dir)
            student_model.load_state_dict(state_read['model_state'])
            optimizer.load_state_dict(state_read['optim_state'])
            epoch = state_read['Epoch']
            print("retaining")

    # Train
    while epoch < epoch_num:
        import datetime
        start = datetime.datetime.now()

        for batch_idx, (multi_sample, single_sample, label) in enumerate(
                tqdm(train_loader, desc="Epoch {}/{}".format(epoch, epoch_num))):

            # if epoch == 0:
            #     continue

            data_read_time = (datetime.datetime.now() - start)
            # print("data_read_time:", data_read_time.total_seconds())
            start = datetime.datetime.now()
            batch_num += 1

            img_rgb, img_ir, img_depth, target = multi_sample['image_x'], multi_sample['image_ir'], \
                multi_sample['image_depth'], multi_sample[
                'binary_label']

            if torch.cuda.is_available():
                img_rgb = img_rgb.cuda()
                img_ir = img_ir.cuda()
                img_depth = img_depth.cuda()
                target = target.cuda()
                single_sample = single_sample.cuda()
                label = label.cuda()

            optimizer.zero_grad()

            teacher_whole_out, teacher_patch_out, teacher_patch_strength = teacher_model(img_rgb, img_ir, img_depth)

            if args.student_data == 'multi_rgb':
                student_whole_out, student_patch_out, student_patch_strength = student_model(img_rgb)
            else:
                student_whole_out, student_patch_out, student_patch_strength = student_model(single_sample)

            time_forward = datetime.datetime.now() - start
            # print("time_forward:", time_forward.total_seconds())

            # 蒸馏损失
            if args.kd_mode in ['logits', 'st', 'multi_st']:
                # patch_loss = mmd_loss(student_patch_out, teacher_patch_out.detach())
                # patch_loss = patch_loss.cuda()
                # whole_loss = criterionKD(student_whole_out, teacher_whole_out.detach())
                # whole_loss = whole_loss.cuda()
                # kd_loss = patch_loss + whole_loss
                # kd_loss = mmd_loss(student_patch_out, teacher_patch_out.detach())

                # multi kd/mmd
                # student_patch_out = torch.flatten(student_patch_out, start_dim=1,end_dim=2)
                # teacher_patch_out = torch.flatten(teacher_patch_out,start_dim=1,end_dim=2)
                # kd_loss = mmd_loss(student_patch_out, teacher_patch_out.detach())
                # print(teacher_patch_strength)

                # weight = torch.mean(student_patch_strength, dim=0)
                # weight = weight.cuda()
                # print(weight.shape)
                if args.weight_patch:
                    teacher_patch_strength = teacher_patch_strength.mean(dim=0)
                    kd_loss = criterionKD(student_patch_out, teacher_patch_out.detach(), weight=teacher_patch_strength)
                else:
                    kd_loss = criterionKD(student_patch_out, teacher_patch_out.detach(), weight=None)

                kd_loss = kd_loss.cuda()
                # print(kd_loss.is_cuda)
            else:
                kd_loss = 0
                print("kd_Loss error")

            # 分类损失
            if args.student_data == 'multi_rgb':
                cls_loss = criterionCls(student_whole_out, target)
            else:
                cls_loss = criterionCls(student_whole_out, label)

            cls_loss = cls_loss.cuda()

            loss = cls_loss + kd_loss * args.lambda_kd
            # print(loss.is_cuda)

            train_loss += loss.item()
            cls_loss_sum += cls_loss.item()
            kd_loss_sum += kd_loss.item()
            loss.backward()
            optimizer.step()
            # if batch_idx % log_interval == 0:  # 准备打印相关信息，args.log_interval是最开头设置的好了的参数
            #     print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            #         epoch, batch_idx * len(data), len(train_loader.dataset),
            #                100. * batch_idx / len(train_loader), loss.item()))

        # testing
        result_test = calc_accuracy_kd_patch(model=student_model, args=args, loader=test_loader, hter=True)
        print(result_test)
        accuracy_test = result_test[0]
        hter_test = result_test[3]
        acer_test = result_test[-1]

        if acer_test < acer_best and epoch > 0:
            acer_best = acer_test
            save_path = os.path.join(args.model_root, args.name + '_acer_best_' + '.pth')
            torch.save(student_model.state_dict(), save_path, )

        if hter_test < hter_best and epoch > 0:
            hter_best = hter_test
            save_path = os.path.join(args.model_root, args.name + '_hter_best_' + '.pth')
            torch.save(student_model.state_dict(), save_path, )

        if accuracy_test > accuracy_best and epoch > 0:
            accuracy_best = accuracy_test
            save_path = os.path.join(args.model_root, args.name + '.pth')
            torch.save(student_model.state_dict(), save_path, )
        log_list.append(train_loss / len(train_loader))
        log_list.append(accuracy_test)
        log_list.append(accuracy_best)
        print(cls_loss_sum / len(train_loader), kd_loss_sum / (len(train_loader)))

        print(
            "Epoch {}, loss={:.5f}, accuracy_test={:.5f},  accuracy_best={:.5f}".format(epoch,
                                                                                        train_loss / len(
                                                                                            train_loader),
                                                                                        accuracy_test, accuracy_best))
        train_loss = 0
        cls_loss_sum = 0
        kd_loss_sum = 0

        if args.lr_decrease:
            if args.lr_warmup:
                scheduler_warmup.step(epoch=epoch)
            else:
                cos_scheduler.step(epoch=epoch)
        if epoch < 20:
            print(epoch, optimizer.param_groups[0]['lr'])

        # save model and para
        if epoch % save_interval == 0:
            train_state = {
                "Epoch": epoch,
                "model_state": student_model.state_dict(),
                "optim_state": optimizer.state_dict(),
                "args": args
            }
            models_dir = args.model_root + '/' + args.name + '.pt'
            torch.save(train_state, models_dir)

        #  save log
        with open(log_dir, 'a+', newline='') as f:
            # 训练结果
            my_writer = csv.writer(f)
            my_writer.writerow(log_list)
            log_list = []
        epoch = epoch + 1
    train_duration_sec = int(time.time() - start)
    print("training is end", train_duration_sec)


def train_knowledge_distill_patch_cefa(net_dict, cost_dict, optimizer, train_loader, test_loader, args):
    '''

    :param model:
    :param cost:
    :param optimizer:
    :param train_loader:
    :param test_loader:
    :param args:
    :return:
    '''

    print(args)
    if not os.path.exists(args.model_root):
        os.makedirs(args.model_root)
    if not os.path.exists(args.log_root):
        os.makedirs(args.log_root)

    models_dir = args.model_root + '/' + args.name + '.pt'
    log_dir = args.log_root + '/' + args.name + '.csv'

    # save args
    with open(log_dir, 'a+', newline='') as f:
        my_writer = csv.writer(f)
        args_dict = vars(args)
        for key, value in args_dict.items():
            my_writer.writerow([key, value])
        f.close()

    student_model = net_dict['snet']
    teacher_model = net_dict['tnet']

    criterionCls = cost_dict['criterionCls']
    criterionKD = cost_dict['criterionKD']

    from loss.kd.pkt import PKTCosSim

    from loss.kd.st import SoftTarget

    if torch.cuda.is_available():
        mmd_loss = MMD_loss().cuda()
    else:
        mmd_loss = MMD_loss()
    if torch.cuda.is_available():
        bce_loss = nn.BCELoss().cuda()
    else:
        bce_loss = nn.BCELoss()

    if torch.cuda.is_available():
        pkd_loss = PKTCosSim().cuda()
    else:
        pkd_loss = PKTCosSim()

    if torch.cuda.is_available():
        soft_t_loss = SoftTarget(T=2).cuda()
    else:
        soft_t_loss = SoftTarget(T=2)

    #  learning rate decay
    if args.lr_decrease == 'cos':
        print("lrcos is using")
        cos_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.train_epoch + 20, eta_min=0)

        if args.lr_warmup:
            scheduler_warmup = GradualWarmupScheduler(args, optimizer, multiplier=1,
                                                      after_scheduler=cos_scheduler)
    elif args.lr_decrease == 'multi_step':
        print("multi_step is using")
        cos_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[np.int(args.train_epoch * 1 / 6),
                                                                              np.int(args.train_epoch * 2 / 6),
                                                                              np.int(args.train_epoch * 3 / 6)])
        if args.lr_warmup:
            scheduler_warmup = GradualWarmupScheduler(args, optimizer, multiplier=1,
                                                      after_scheduler=cos_scheduler)

    # Training initialization
    epoch_num = args.train_epoch
    log_interval = args.log_interval
    save_interval = args.save_interval
    batch_num = 0
    train_loss = 0
    cls_loss_sum = 0
    kd_loss_sum = 0
    epoch = 0
    accuracy_best = 0
    acer_best = 1
    hter_best = 1
    log_list = []  # log need to save

    if args.retrain:
        if not os.path.exists(models_dir):
            print("no trained model")
        else:
            state_read = torch.load(models_dir)
            student_model.load_state_dict(state_read['model_state'])
            optimizer.load_state_dict(state_read['optim_state'])
            epoch = state_read['Epoch']
            print("retaining")

    # Train
    while epoch < epoch_num:
        import datetime
        start = datetime.datetime.now()

        for batch_idx, multi_sample in enumerate(
                tqdm(train_loader, desc="Epoch {}/{}".format(epoch, epoch_num))):

            # if epoch == 0:
            #     continue

            data_read_time = (datetime.datetime.now() - start)
            # print("data_read_time:", data_read_time.total_seconds())
            start = datetime.datetime.now()
            batch_num += 1

            img_rgb, img_ir, img_depth, target = multi_sample['image_x'], multi_sample['image_ir'], \
                multi_sample['image_depth'], multi_sample[
                'binary_label']

            if torch.cuda.is_available():
                img_rgb = img_rgb.cuda()
                img_ir = img_ir.cuda()
                img_depth = img_depth.cuda()
                target = target.cuda()
            label = target

            optimizer.zero_grad()

            teacher_whole_out, teacher_patch_out, teacher_patch_strength = teacher_model(img_rgb, img_ir, img_depth)

            student_whole_out, student_patch_out, student_patch_strength = student_model(img_rgb)

            # print("time_forward:", time_forward.total_seconds())

            # 蒸馏损失
            if args.kd_mode in ['logits', 'st', 'multi_st']:
                # patch_loss = mmd_loss(student_patch_out, teacher_patch_out.detach())
                # patch_loss = patch_loss.cuda()
                # whole_loss = criterionKD(student_whole_out, teacher_whole_out.detach())
                # whole_loss = whole_loss.cuda()
                # kd_loss = patch_loss + whole_loss
                # kd_loss = mmd_loss(student_patch_out, teacher_patch_out.detach())

                # multi kd/mmd
                # student_patch_out = torch.flatten(student_patch_out, start_dim=1,end_dim=2)
                # teacher_patch_out = torch.flatten(teacher_patch_out,start_dim=1,end_dim=2)
                # kd_loss = mmd_loss(student_patch_out, teacher_patch_out.detach())
                # print(teacher_patch_strength)

                # weight = torch.mean(student_patch_strength, dim=0)
                # weight = weight.cuda()
                # print(weight.shape)
                if args.select:
                    kd_loss = criterionKD(student_patch_out, teacher_patch_out.detach(), weight=student_patch_strength)
                else:
                    kd_loss = criterionKD(student_patch_out, teacher_patch_out.detach(), weight=None)

                # classic KD
                # kd_loss = soft_t_loss(student_whole_out, teacher_whole_out.detach())

                # kd_loss = kd_loss.cuda()
                # print(kd_loss.is_cuda)
            else:
                kd_loss = 0
                print("kd_Loss error")

            # 分类损失
            if args.student_data == 'multi_rgb':
                cls_loss = criterionCls(student_whole_out, target)
            else:
                cls_loss = criterionCls(student_whole_out, label)

            cls_loss = cls_loss.cuda()

            loss = cls_loss + kd_loss * args.lambda_kd
            # print(loss.is_cuda)

            train_loss += loss.item()
            cls_loss_sum += cls_loss.item()
            kd_loss_sum += kd_loss.item()
            loss.backward()
            optimizer.step()
            # if batch_idx % log_interval == 0:  # 准备打印相关信息，args.log_interval是最开头设置的好了的参数
            #     print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            #         epoch, batch_idx * len(data), len(train_loader.dataset),
            #                100. * batch_idx / len(train_loader), loss.item()))

        # testing
        result_test = calc_accuracy_kd_patch(model=student_model, args=args, loader=test_loader, hter=True)
        print(result_test)
        accuracy_test = result_test[0]
        hter_test = result_test[3]
        acer_test = result_test[-1]

        if acer_test < acer_best and epoch > 0:
            acer_best = acer_test
            save_path = os.path.join(args.model_root, args.name + '_acer_best_' + '.pth')
            torch.save(student_model.state_dict(), save_path, )

        if hter_test < hter_best and epoch > 0:
            hter_best = hter_test
            save_path = os.path.join(args.model_root, args.name + '_hter_best_' + '.pth')
            torch.save(student_model.state_dict(), save_path, )

        if accuracy_test > accuracy_best and epoch > 0:
            accuracy_best = accuracy_test
            save_path = os.path.join(args.model_root, args.name + '.pth')
            torch.save(student_model.state_dict(), save_path, )
        log_list.append(train_loss / len(train_loader))
        log_list.append(accuracy_test)
        log_list.append(accuracy_best)
        print(cls_loss_sum / len(train_loader), kd_loss_sum / (len(train_loader)))

        print(
            "Epoch {}, loss={:.5f}, accuracy_test={:.5f},  accuracy_best={:.5f}".format(epoch,
                                                                                        train_loss / len(
                                                                                            train_loader),
                                                                                        accuracy_test, accuracy_best))
        train_loss = 0
        cls_loss_sum = 0
        kd_loss_sum = 0

        if args.lr_decrease:
            if args.lr_warmup:
                scheduler_warmup.step(epoch=epoch)
            else:
                cos_scheduler.step(epoch=epoch)
        if epoch < 20:
            print(epoch, optimizer.param_groups[0]['lr'])

        # save model and para
        if epoch % save_interval == 0:
            train_state = {
                "Epoch": epoch,
                "model_state": student_model.state_dict(),
                "optim_state": optimizer.state_dict(),
                "args": args
            }
            models_dir = args.model_root + '/' + args.name + '.pt'
            torch.save(train_state, models_dir)

        #  save log
        with open(log_dir, 'a+', newline='') as f:
            # 训练结果
            my_writer = csv.writer(f)
            my_writer.writerow(log_list)
            log_list = []
        epoch = epoch + 1
    train_duration_sec = int(time.time() - start)
    print("training is end", train_duration_sec)


def train_knowledge_distill_patch_feature_auxi(net_dict, cost_dict, optimizer, train_loader, test_loader, args):
    '''

    :param model:
    :param cost:
    :param optimizer:
    :param train_loader:
    :param test_loader:
    :param args:
    :return:
    '''
    from loss.kd.pkt import PKTCosSim
    from loss.kd.sp import SP, DAD
    from loss.kd.at import AT
    print(args)
    if not os.path.exists(args.model_root):
        os.makedirs(args.model_root)
    if not os.path.exists(args.log_root):
        os.makedirs(args.log_root)

    models_dir = args.model_root + '/' + args.name + '.pt'
    log_dir = args.log_root + '/' + args.name + '.csv'

    # save args
    with open(log_dir, 'a+', newline='') as f:
        my_writer = csv.writer(f)
        args_dict = vars(args)
        for key, value in args_dict.items():
            my_writer.writerow([key, value])
        f.close()

    auxi_cross_entropy = nn.CrossEntropyLoss(reduction='none')

    student_model = net_dict['snet']
    teacher_model = net_dict['tnet']

    criterionCls = cost_dict['criterionCls']
    criterionKD = cost_dict['criterionKD']

    if torch.cuda.is_available():
        mmd_loss = MMD_loss().cuda()
    else:
        mmd_loss = MMD_loss()

    if torch.cuda.is_available():
        dad_loss = DAD().cuda()
    else:
        dad_loss = DAD()

    if torch.cuda.is_available():
        sp_loss = SP().cuda()
    else:
        sp_loss = SP()

    if torch.cuda.is_available():
        at_loss = AT(p=2).cuda()
    else:
        at_loss = AT(p=2)

    if torch.cuda.is_available():
        bce_loss = nn.BCELoss().cuda()
    else:
        bce_loss = nn.BCELoss()

    if torch.cuda.is_available():
        mse_loss = nn.MSELoss().cuda()
    else:
        mse_loss = nn.MSELoss()

    #  learning rate decay
    if args.lr_decrease == 'cos':
        print("lrcos is using")
        cos_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.train_epoch + 20, eta_min=0)

        if args.lr_warmup:
            scheduler_warmup = GradualWarmupScheduler(args, optimizer, multiplier=1,
                                                      after_scheduler=cos_scheduler)
    elif args.lr_decrease == 'multi_step':
        print("multi_step is using")
        cos_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[np.int(args.train_epoch * 1 / 6),
                                                                              np.int(args.train_epoch * 2 / 6),
                                                                              np.int(args.train_epoch * 3 / 6)])
        if args.lr_warmup:
            scheduler_warmup = GradualWarmupScheduler(args, optimizer, multiplier=1,
                                                      after_scheduler=cos_scheduler)

    # Training initialization
    epoch_num = args.train_epoch
    log_interval = args.log_interval
    save_interval = args.save_interval
    batch_num = 0
    train_loss = 0
    cls_loss_sum = 0
    kd_logits_loss_sum = 0
    kd_feature_loss_sum = 0
    epoch = 0
    accuracy_best = 0
    acer_best = 1
    hter_best = 1
    log_list = []  # log need to save

    if args.retrain:
        if not os.path.exists(models_dir):
            print("no trained model")
        else:
            state_read = torch.load(models_dir)
            student_model.load_state_dict(state_read['model_state'])
            optimizer.load_state_dict(state_read['optim_state'])
            epoch = state_read['Epoch']
            print("retaining")

    # Train
    while epoch < epoch_num:
        start = datetime.datetime.now()

        for batch_idx, batch_sample in enumerate(
                tqdm(train_loader, desc="Epoch {}/{}".format(epoch, epoch_num))):

            batch_num += 1
            img_rgb, img_ir, img_depth, target = batch_sample['image_x'], batch_sample['image_ir'], \
                batch_sample['image_depth'], batch_sample[
                'binary_label']
            if epoch == 0:
                continue

            data_read_time = (datetime.datetime.now() - start)

            if torch.cuda.is_available():
                img_rgb = img_rgb.cuda()
                img_ir = img_ir.cuda()
                img_depth = img_depth.cuda()
                target = target.cuda()

            optimizer.zero_grad()

            teacher_whole_out, teacher_layer3, teacher_layer4 = teacher_model(img_rgb, img_ir, img_depth)

            student_whole_out, student_layer3, student_layer4, x_rgb_out, x_ir_out, x_depth_out, p = student_model(
                img_rgb, img_ir, img_depth)

            time_forward = datetime.datetime.now() - start
            # print("time_forward:", time_forward.total_seconds())

            # logits蒸馏损失
            # if args.kd_mode in ['logits', 'st']:
            #     # patch_loss = mmd_loss(student_patch_out, teacher_patch_out.detach())
            #     # patch_loss = patch_loss.cuda()
            #     # whole_loss = criterionKD(student_whole_out, teacher_whole_out.detach())
            #     # whole_loss = whole_loss.cuda()
            #     # kd_loss = patch_loss + whole_loss
            #     kd_logits_loss = mmd_loss(student_patch_out, teacher_patch_out.detach())
            #     # kd_logits_loss = bce_loss(student_patch_out, teacher_patch_out.detach())
            #     kd_logits_loss = kd_logits_loss.cuda()
            # else:
            #     kd_logits_loss = 0
            #     print("kd_Loss error")

            # feature 蒸馏损失
            # student_layer3 = torch.mean(student_layer3, dim=1)
            # teacher_layer3 = torch.mean(teacher_layer3, dim=1)
            # kd_feature_loss = mse_loss(student_layer3, teacher_layer3)

            # student_layer3=torch.unsqueeze(student_layer3,dim=1)
            kd_feature_loss_1 = sp_loss(student_layer3, teacher_layer3)
            kd_feature_loss_2 = sp_loss(student_layer4, teacher_layer4)
            kd_feature_loss = kd_feature_loss_2

            # 分类损失

            fusion_loss = criterionCls(student_whole_out, target)

            x_rgb_loss_batch = auxi_cross_entropy(x_rgb_out, target)

            x_rgb_loss = torch.sum(x_rgb_loss_batch * ((p[:, 0]) * (1 - p[:, 1]) * (1 - p[:, 2]))) / p.shape[0]

            x_ir_loss_batch = auxi_cross_entropy(x_ir_out, target)
            x_ir_loss = torch.sum(x_ir_loss_batch * ((p[:, 1]) * (1 - p[:, 0]) * (1 - p[:, 2]))) / p.shape[0]

            x_depth_loss_batch = auxi_cross_entropy(x_depth_out, target)
            x_depth_loss = torch.sum(x_depth_loss_batch * ((p[:, 2]) * (1 - p[:, 0]) * (1 - p[:, 1]))) / p.shape[0]

            cls_loss = fusion_loss + x_rgb_loss + x_depth_loss + x_ir_loss

            cls_loss = cls_loss.cuda()

            loss = cls_loss + kd_feature_loss * args.lambda_kd_feature

            train_loss += loss.item()
            cls_loss_sum += cls_loss.item()
            # kd_logits_loss_sum += kd_logits_loss.item()
            kd_feature_loss_sum += kd_feature_loss.item()
            loss.backward()
            optimizer.step()
            # if batch_idx % log_interval == 0:  # 准备打印相关信息，args.log_interval是最开头设置的好了的参数
            #     print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            #         epoch, batch_idx * len(data), len(train_loader.dataset),
            #                100. * batch_idx / len(train_loader), loss.item()))

        # testing
        result_test = calc_accuracy_kd_patch_feature(model=student_model, args=args, loader=test_loader, hter=True)
        print(result_test)
        accuracy_test = result_test[0]

        hter_test = result_test[3]
        acer_test = result_test[-1]

        if acer_test < acer_best and epoch > 15:
            acer_best = acer_test
            save_path = os.path.join(args.model_root, args.name + '_acer_best_' + '.pth')
            torch.save(student_model.state_dict(), save_path, )

        if hter_test < hter_best and epoch > 15:
            hter_best = hter_test
            save_path = os.path.join(args.model_root, args.name + '_hter_best_' + '.pth')
            torch.save(student_model.state_dict(), save_path, )

        if accuracy_test > accuracy_best and epoch > 15:
            accuracy_best = accuracy_test
            save_path = os.path.join(args.model_root, args.name + '.pth')
            torch.save(student_model.state_dict(), save_path, )
        log_list.append(train_loss / len(train_loader))
        log_list.append(accuracy_test)
        log_list.append(accuracy_best)
        print(cls_loss_sum / len(train_loader), kd_logits_loss_sum / (len(train_loader)),
              kd_feature_loss_sum / (len(train_loader)))

        print(
            "Epoch {}, loss={:.5f}, accuracy_test={:.5f},  accuracy_best={:.5f}".format(epoch,
                                                                                        train_loss / len(
                                                                                            train_loader),
                                                                                        accuracy_test, accuracy_best))
        train_loss = 0
        cls_loss_sum = 0
        kd_feature_loss_sum = 0
        kd_logits_loss_sum = 0

        if args.lr_decrease:
            if args.lr_warmup:
                scheduler_warmup.step(epoch=epoch)
            else:
                cos_scheduler.step(epoch=epoch)
        if epoch < 20:
            print(epoch, optimizer.param_groups[0]['lr'])

        # save model and para
        if epoch % save_interval == 0:
            train_state = {
                "Epoch": epoch,
                "model_state": student_model.state_dict(),
                "optim_state": optimizer.state_dict(),
                "args": args
            }
            models_dir = args.model_root + '/' + args.name + '.pt'
            torch.save(train_state, models_dir)

        #  save log
        with open(log_dir, 'a+', newline='') as f:
            # 训练结果
            my_writer = csv.writer(f)
            my_writer.writerow(log_list)
            log_list = []
        epoch = epoch + 1
    # train_duration_sec = int(time.time() - start)
    # print("training is end", train_duration_sec)


def train_knowledge_distill_patch_feature_auxi_weak(net_dict, cost_dict, optimizer, train_loader, test_loader, args):
    '''

    :param model:
    :param cost:
    :param optimizer:
    :param train_loader:
    :param test_loader:
    :param args:
    :return:
    '''
    from loss.kd.pkt import PKTCosSim
    from loss.kd.sp import SP, DAD
    from loss.kd.at import AT
    print(args)
    if not os.path.exists(args.model_root):
        os.makedirs(args.model_root)
    if not os.path.exists(args.log_root):
        os.makedirs(args.log_root)

    models_dir = args.model_root + '/' + args.name + '.pt'
    log_dir = args.log_root + '/' + args.name + '.csv'

    # save args
    with open(log_dir, 'a+', newline='') as f:
        my_writer = csv.writer(f)
        args_dict = vars(args)
        for key, value in args_dict.items():
            my_writer.writerow([key, value])
        f.close()

    auxi_cross_entropy = nn.CrossEntropyLoss(reduction='none')

    student_model = net_dict['snet']
    teacher_model = net_dict['tnet']

    criterionCls = cost_dict['criterionCls']
    criterionKD = cost_dict['criterionKD']

    if torch.cuda.is_available():
        mmd_loss = MMD_loss().cuda()
    else:
        mmd_loss = MMD_loss()

    if torch.cuda.is_available():
        dad_loss = DAD().cuda()
    else:
        dad_loss = DAD()

    if torch.cuda.is_available():
        sp_loss = SP().cuda()
    else:
        sp_loss = SP()

    if torch.cuda.is_available():
        at_loss = AT(p=2).cuda()
    else:
        at_loss = AT(p=2)

    if torch.cuda.is_available():
        bce_loss = nn.BCELoss().cuda()
    else:
        bce_loss = nn.BCELoss()

    if torch.cuda.is_available():
        mse_loss = nn.MSELoss().cuda()
    else:
        mse_loss = nn.MSELoss()

    #  learning rate decay
    if args.lr_decrease == 'cos':
        print("lrcos is using")
        cos_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.train_epoch + 20, eta_min=0)

        if args.lr_warmup:
            scheduler_warmup = GradualWarmupScheduler(args, optimizer, multiplier=1,
                                                      after_scheduler=cos_scheduler)
    elif args.lr_decrease == 'multi_step':
        print("multi_step is using")
        cos_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[np.int(args.train_epoch * 1 / 6),
                                                                              np.int(args.train_epoch * 2 / 6),
                                                                              np.int(args.train_epoch * 3 / 6)])
        if args.lr_warmup:
            scheduler_warmup = GradualWarmupScheduler(args, optimizer, multiplier=1,
                                                      after_scheduler=cos_scheduler)

    # Training initialization
    epoch_num = args.train_epoch
    log_interval = args.log_interval
    save_interval = args.save_interval
    batch_num = 0
    train_loss = 0
    cls_loss_sum = 0
    kd_logits_loss_sum = 0
    kd_feature_loss_sum = 0
    epoch = 0
    accuracy_best = 0
    acer_best = 1
    hter_best = 1
    log_list = []  # log need to save

    if args.retrain:
        if not os.path.exists(models_dir):
            print("no trained model")
        else:
            state_read = torch.load(models_dir)
            student_model.load_state_dict(state_read['model_state'])
            optimizer.load_state_dict(state_read['optim_state'])
            epoch = state_read['Epoch']
            print("retaining")

    # Train

    modality_combination = [[1, 1, 0], [1, 0, 1], [0, 1, 1], [1, 1, 1]]
    disttribution_distance = [0] * (len(modality_combination) - 1)

    while epoch < epoch_num:
        student_model.p = [0, 0, 0]
        start = datetime.datetime.now()
        fusion_loss_weak_list = []
        fuse_loss_strong_list = []
        for batch_idx, batch_sample in enumerate(
                tqdm(train_loader, desc="Epoch {}/{}".format(epoch, epoch_num))):

            batch_num += 1
            img_rgb, img_ir, img_depth, target = batch_sample['image_x'], batch_sample['image_ir'], \
                batch_sample['image_depth'], batch_sample[
                'binary_label']
            # if epoch == 0:
            #     continue

            data_read_time = (datetime.datetime.now() - start)

            if torch.cuda.is_available():
                img_rgb = img_rgb.cuda()
                img_ir = img_ir.cuda()
                img_depth = img_depth.cuda()
                target = target.cuda()

            optimizer.zero_grad()

            teacher_whole_out, teacher_layer3, teacher_layer4 = teacher_model(img_rgb, img_ir, img_depth)

            student_whole_out, student_layer3, student_layer4, x_rgb_out, x_ir_out, x_depth_out, p = student_model(
                img_rgb, img_ir, img_depth)

            time_forward = datetime.datetime.now() - start
            # print("time_forward:", time_forward.total_seconds())

            # logits蒸馏损失
            # if args.kd_mode in ['logits', 'st']:
            #     # patch_loss = mmd_loss(student_patch_out, teacher_patch_out.detach())
            #     # patch_loss = patch_loss.cuda()
            #     # whole_loss = criterionKD(student_whole_out, teacher_whole_out.detach())
            #     # whole_loss = whole_loss.cuda()
            #     # kd_loss = patch_loss + whole_loss
            #     kd_logits_loss = mmd_loss(student_patch_out, teacher_patch_out.detach())
            #     # kd_logits_loss = bce_loss(student_patch_out, teacher_patch_out.detach())
            #     kd_logits_loss = kd_logits_loss.cuda()
            # else:
            #     kd_logits_loss = 0
            #     print("kd_Loss error")

            # feature 蒸馏损失
            # student_layer3 = torch.mean(student_layer3, dim=1)
            # teacher_layer3 = torch.mean(teacher_layer3, dim=1)
            # kd_feature_loss = mse_loss(student_layer3, teacher_layer3)

            # student_layer3=torch.unsqueeze(student_layer3,dim=1)
            kd_feature_loss_1 = sp_loss(student_layer3, teacher_layer3)
            kd_feature_loss_2 = sp_loss(student_layer4, teacher_layer4)
            kd_feature_loss = kd_feature_loss_2

            # print(kd_feature_loss.shape)

            teacher_whole_out_prob = F.softmax(teacher_whole_out, dim=1)
            H_teacher = torch.sum(-teacher_whole_out_prob * torch.log(teacher_whole_out_prob), dim=1)
            # print(H_teacher.shape)
            # H_teacher_prob = F.softmax(H_teacher * 64, dim=0)
            H_teacher_prob = H_teacher / torch.sum(H_teacher)
            kd_feature_loss = torch.sum(kd_feature_loss * H_teacher_prob)

            # 分类损失

            if args.dataset == 'surf':

                fusion_loss = criterionCls(student_whole_out, target)

                x_rgb_loss_batch = auxi_cross_entropy(x_rgb_out, target)

                # x_auxi_weak = torch.sum(x_rgb_loss_batch * ((1 - p[:, 1]))) / p.shape[0]

                x_rgb_loss = torch.sum(x_rgb_loss_batch * ((p[:, 0]) * (1 - p[:, 1]) * (1 - p[:, 2]))) / p.shape[0]

                x_ir_loss_batch = auxi_cross_entropy(x_ir_out, target)

                # print(type(p[:, 0]))
                index = p[:, 0].int() | p[:, 2].int()
                index = index.float()
                # print(index)
                x_ir_loss = torch.sum(x_ir_loss_batch * ((p[:, 0]) * (1 - p[:, 1]) * (p[:, 2]))) / p.shape[0]

                x_depth_loss_batch = auxi_cross_entropy(x_depth_out, target)
                x_depth_loss = torch.sum(x_depth_loss_batch * ((p[:, 2]) * (1 - p[:, 0]) * (1 - p[:, 1]))) / p.shape[0]

            else:

                fusion_loss = criterionCls(student_whole_out, target)

                x_rgb_loss_batch = auxi_cross_entropy(x_rgb_out, target)

                # x_auxi_weak = torch.sum(x_rgb_loss_batch * ((1 - p[:, 0]))) / p.shape[0]
                x_rgb_loss = torch.sum(x_rgb_loss_batch * ((1 - p[:, 0]) * (p[:, 1]) * (1 - p[:, 2]))) / p.shape[0]

                x_ir_loss_batch = auxi_cross_entropy(x_ir_out, target)

                # print(type(p[:, 0]))
                index = p[:, 0].int() | p[:, 2].int()
                # index = index.float()
                # print(index)
                x_ir_loss = torch.sum(x_ir_loss_batch * ((1 - p[:, 2]) * (p[:, 1]) * (p[:, 1]))) / p.shape[0]

                x_depth_loss_batch = auxi_cross_entropy(x_depth_out, target)
                x_depth_loss = torch.sum(x_depth_loss_batch * ((p[:, 1]) * (1 - p[:, 2]) * (1 - p[:, 0]))) / p.shape[0]

            if epoch > args.begin_epoch:
                cls_loss = fusion_loss + x_rgb_loss + x_depth_loss + x_ir_loss
                # cls_loss = fusion_loss + x_auxi_weak

            else:
                cls_loss = fusion_loss

            # print(p.cpu().detach().numpy())
            # c=(p[:, 0]) * (1 - p[:, 1]) * (p[:, 2])
            # print(c[1:10],p[1:10, 1])
            fusion_loss_weak = torch.sum(fusion_loss * ((1 - p[:, 1])))
            fuse_loss_strong = torch.sum(fusion_loss * p[:, 1])

            if epoch == 4:
                fusion_loss_weak_list.append(fusion_loss_weak.cpu().detach().numpy())
                fuse_loss_strong_list.append(fuse_loss_strong.cpu().detach().numpy())

            cls_loss = cls_loss.cuda()
            if epoch > 5:
                loss = cls_loss + kd_feature_loss * args.lambda_kd_feature
            else:
                loss = cls_loss

            train_loss += loss.item()
            cls_loss_sum += cls_loss.item()
            # kd_logits_loss_sum += kd_logits_loss.item()
            kd_feature_loss_sum += kd_feature_loss.item()
            loss.backward()
            optimizer.step()
            # if batch_idx % log_interval == 0:  # 准备打印相关信息，args.log_interval是最开头设置的好了的参数
            #     print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            #         epoch, batch_idx * len(data), len(train_loader.dataset),
            #                100. * batch_idx / len(train_loader), loss.item()))

        # testing

        if epoch == 4:
            with open('weak_strong_loss.txt', 'a+', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(fusion_loss_weak_list)
                writer.writerow(fuse_loss_strong_list)

        student_model.p = [0, 0, 0]
        result_test, _ = calc_accuracy_kd_patch_feature(model=student_model, args=args, loader=test_loader, hter=True)
        print(result_test)
        accuracy_test = result_test[0]

        print(fusion_loss.cpu().detach().numpy(), (x_rgb_loss + x_depth_loss + x_ir_loss).cpu().detach().numpy(),
              kd_feature_loss.cpu().detach().numpy())

        hter_test = result_test[3]
        acer_test = result_test[-1]

        if epoch <= 5:
            # root_dir = "../data/CASIA-SURF"
            # txt_dir = root_dir + '/test_private_list.txt'
            # surf_dataset = SURF(txt_dir=txt_dir,
            #                     root_dir=root_dir,
            #                     transform=surf_multi_transforms_test, miss_modal=args.miss_modal, times=1)
            #
            # surf_dataset = SURF_generate(rgb_dir=args.rgb_root, depth_dir=args.depth_root, ir_dir=args.ir_root,
            #                              transform=surf_multi_transforms_test)

            # test_loader_t = torch.utils.data.DataLoader(
            #     dataset=surf_dataset,
            #     batch_size=64,
            #     shuffle=False,
            #     num_workers=4)
            label_disttribution = []
            for j in range(len(modality_combination)):
                args.p = modality_combination[j]
                print(args.p)
                student_model.p = modality_combination[j]
                result_test, label_predict = calc_accuracy_kd_patch_feature(model=student_model, args=args,
                                                                            loader=test_loader,
                                                                            hter=True)
                label_predict_hist = get_dataself_hist(np.array(label_predict.cpu()))
                print(result_test)
                print(label_predict_hist)
                v_list = [0] * args.class_num
                for k, v in label_predict_hist.items():
                    v_list[int(k)] = v
                v_arr = np.array(v_list)
                v_arr = v_arr / (np.sum(v_arr))
                label_disttribution.append([list(v_arr)])

            # print(label_disttribution)
            label_disttribution = torch.tensor(label_disttribution).float()
            if epoch > 1:
                for i in range(len(label_disttribution) - 1):
                    distance = F.kl_div(
                        F.log_softmax(label_disttribution[len(label_disttribution) - 1], dim=1),
                        F.softmax(label_disttribution[i], dim=1), reduction='batchmean')
                    print(distance)
                    disttribution_distance[i] = disttribution_distance[i] + 0.2 * (distance - disttribution_distance[i])

                # label_predict=np.array(label_predict.cpu())
                # label_disttribution.append(label_predict)
            # if epoch > 1:
            #     for i in range(len(label_disttribution) - 1):
            # disttribution_distance[i] = disttribution_distance[i] + 0.2 * (np.sum(label_disttribution[len(label_disttribution)-1]!=label_disttribution[i]))

            print(np.array(disttribution_distance))

        if acer_test < acer_best and epoch > 15:
            acer_best = acer_test
            save_path = os.path.join(args.model_root, args.name + '_acer_best_' + '.pth')
            torch.save(student_model.state_dict(), save_path, )

        if hter_test < hter_best and epoch > 15:
            hter_best = hter_test
            save_path = os.path.join(args.model_root, args.name + '_hter_best_' + '.pth')
            torch.save(student_model.state_dict(), save_path, )

        if accuracy_test > accuracy_best and epoch > 15:
            accuracy_best = accuracy_test
            save_path = os.path.join(args.model_root, args.name + '.pth')
            torch.save(student_model.state_dict(), save_path, )
        log_list.append(train_loss / len(train_loader))
        log_list.append(accuracy_test)
        log_list.append(accuracy_best)
        print(cls_loss_sum / len(train_loader), kd_logits_loss_sum / (len(train_loader)),
              kd_feature_loss_sum / (len(train_loader)))

        print(
            "Epoch {}, loss={:.5f}, accuracy_test={:.5f},  accuracy_best={:.5f}".format(epoch,
                                                                                        train_loss / len(
                                                                                            train_loader),
                                                                                        accuracy_test, accuracy_best))
        train_loss = 0
        cls_loss_sum = 0
        kd_feature_loss_sum = 0
        kd_logits_loss_sum = 0

        if args.lr_decrease:
            if args.lr_warmup:
                scheduler_warmup.step(epoch=epoch)
            else:
                cos_scheduler.step(epoch=epoch)
        if epoch < 20:
            print(epoch, optimizer.param_groups[0]['lr'])

        # save model and para
        if epoch % save_interval == 0:
            train_state = {
                "Epoch": epoch,
                "model_state": student_model.state_dict(),
                "optim_state": optimizer.state_dict(),
                "args": args
            }
            models_dir = args.model_root + '/' + args.name + '.pt'
            torch.save(train_state, models_dir)

        #  save log
        with open(log_dir, 'a+', newline='') as f:
            # 训练结果
            my_writer = csv.writer(f)
            my_writer.writerow(log_list)
            log_list = []
        epoch = epoch + 1
    # train_duration_sec = int(time.time() - start)
    # print("training is end", train_duration_sec)


def train_knowledge_distill_patch_feature(net_dict, cost_dict, optimizer, train_loader, test_loader, args):
    '''

    :param model:
    :param cost:
    :param optimizer:
    :param train_loader:
    :param test_loader:
    :param args:
    :return:
    '''
    from loss.kd.pkt import PKTCosSim
    from loss.kd.sp import SP, DAD
    from loss.kd.at import AT
    print(args)
    if not os.path.exists(args.model_root):
        os.makedirs(args.model_root)
    if not os.path.exists(args.log_root):
        os.makedirs(args.log_root)

    models_dir = args.model_root + '/' + args.name + '.pt'
    log_dir = args.log_root + '/' + args.name + '.csv'

    # save args
    with open(log_dir, 'a+', newline='') as f:
        my_writer = csv.writer(f)
        args_dict = vars(args)
        for key, value in args_dict.items():
            my_writer.writerow([key, value])
        f.close()

    student_model = net_dict['snet']
    teacher_model = net_dict['tnet']

    criterionCls = cost_dict['criterionCls']
    criterionKD = cost_dict['criterionKD']

    if torch.cuda.is_available():
        mmd_loss = MMD_loss().cuda()
    else:
        mmd_loss = MMD_loss()

    if torch.cuda.is_available():
        dad_loss = DAD().cuda()
    else:
        dad_loss = DAD()

    if torch.cuda.is_available():
        sp_loss = SP().cuda()
    else:
        sp_loss = SP()

    if torch.cuda.is_available():
        at_loss = AT(p=2).cuda()
    else:
        at_loss = AT(p=2)

    if torch.cuda.is_available():
        bce_loss = nn.BCELoss().cuda()
    else:
        bce_loss = nn.BCELoss()

    if torch.cuda.is_available():
        mse_loss = nn.MSELoss().cuda()
    else:
        mse_loss = nn.MSELoss()

    #  learning rate decay
    if args.lr_decrease == 'cos':
        print("lrcos is using")
        cos_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.train_epoch + 20, eta_min=0)

        if args.lr_warmup:
            scheduler_warmup = GradualWarmupScheduler(args, optimizer, multiplier=1,
                                                      after_scheduler=cos_scheduler)
    elif args.lr_decrease == 'multi_step':
        print("multi_step is using")
        cos_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[np.int(args.train_epoch * 2 / 6),
                                                                              np.int(args.train_epoch * 3 / 6),
                                                                              np.int(args.train_epoch * 4 / 6)])
        if args.lr_warmup:
            scheduler_warmup = GradualWarmupScheduler(args, optimizer, multiplier=1,
                                                      after_scheduler=cos_scheduler)

    # Training initialization
    epoch_num = args.train_epoch
    log_interval = args.log_interval
    save_interval = args.save_interval
    batch_num = 0
    train_loss = 0
    cls_loss_sum = 0
    kd_logits_loss_sum = 0
    kd_feature_loss_sum = 0
    epoch = 0
    accuracy_best = 0
    acer_best = 1
    hter_best = 1
    log_list = []  # log need to save

    if args.retrain:
        if not os.path.exists(models_dir):
            print("no trained model")
        else:
            state_read = torch.load(models_dir)
            student_model.load_state_dict(state_read['model_state'])
            optimizer.load_state_dict(state_read['optim_state'])
            epoch = state_read['Epoch']
            print("retaining")

    # Train
    while epoch < epoch_num:
        start = datetime.datetime.now()

        for batch_idx, batch_sample in enumerate(
                tqdm(train_loader, desc="Epoch {}/{}".format(epoch, epoch_num))):

            batch_num += 1
            img_rgb, img_ir, img_depth, target = batch_sample['image_x'], batch_sample['image_ir'], \
                batch_sample['image_depth'], batch_sample[
                'binary_label']
            if epoch == 0:
                continue

            data_read_time = (datetime.datetime.now() - start)

            if torch.cuda.is_available():
                img_rgb = img_rgb.cuda()
                img_ir = img_ir.cuda()
                img_depth = img_depth.cuda()
                target = target.cuda()

            optimizer.zero_grad()

            teacher_whole_out, teacher_layer3, teacher_layer4 = teacher_model(img_rgb, img_ir, img_depth)

            student_whole_out, student_layer3, student_layer4 = student_model(img_rgb, img_ir, img_depth)

            time_forward = datetime.datetime.now() - start
            # print("time_forward:", time_forward.total_seconds())

            # logits蒸馏损失
            # if args.kd_mode in ['logits', 'st']:
            #     # patch_loss = mmd_loss(student_patch_out, teacher_patch_out.detach())
            #     # patch_loss = patch_loss.cuda()
            #     # whole_loss = criterionKD(student_whole_out, teacher_whole_out.detach())
            #     # whole_loss = whole_loss.cuda()
            #     # kd_loss = patch_loss + whole_loss
            #     kd_logits_loss = mmd_loss(student_patch_out, teacher_patch_out.detach())
            #     # kd_logits_loss = bce_loss(student_patch_out, teacher_patch_out.detach())
            #     kd_logits_loss = kd_logits_loss.cuda()
            # else:
            #     kd_logits_loss = 0
            #     print("kd_Loss error")

            # feature 蒸馏损失
            # student_layer3 = torch.mean(student_layer3, dim=1)
            # teacher_layer3 = torch.mean(teacher_layer3, dim=1)
            # kd_feature_loss = mse_loss(student_layer3, teacher_layer3)

            # student_layer3=torch.unsqueeze(student_layer3,dim=1)
            kd_feature_loss_1 = sp_loss(student_layer3, teacher_layer3)
            kd_feature_loss_2 = sp_loss(student_layer4, teacher_layer4)
            kd_feature_loss = kd_feature_loss_2

            # print(kd_feature_loss.shape)

            # if args.margin:
            #
            #     teacher_whole_out_prob = F.softmax(teacher_whole_out, dim=1)
            #     H_teacher = torch.sum(-teacher_whole_out_prob * torch.log(teacher_whole_out_prob), dim=1)
            #     # print(H_teacher.shape)
            #     # H_teacher_prob = F.softmax(H_teacher * 64, dim=0)
            #     H_teacher_prob = H_teacher / torch.sum(H_teacher)
            #     kd_feature_loss = torch.sum(kd_feature_loss * H_teacher_prob)
            # else:
            #     kd_feature_loss = torch.mean(torch.sum(kd_feature_loss,dim=1))

            # print(H_teacher_prob)

            # 分类损失
            cls_loss = criterionCls(student_whole_out, target)

            cls_loss = cls_loss.cuda()

            loss = cls_loss + kd_feature_loss * args.lambda_kd_feature

            train_loss += loss.item()
            cls_loss_sum += cls_loss.item()
            # kd_logits_loss_sum += kd_logits_loss.item()
            kd_feature_loss_sum += kd_feature_loss.item()
            loss.backward()
            optimizer.step()
            # if batch_idx % log_interval == 0:  # 准备打印相关信息，args.log_interval是最开头设置的好了的参数
            #     print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            #         epoch, batch_idx * len(data), len(train_loader.dataset),
            #                100. * batch_idx / len(train_loader), loss.item()))

        # testing
        result_test, _ = calc_accuracy_kd_patch_feature(model=student_model, args=args, loader=test_loader, hter=True)
        print(result_test)
        accuracy_test = result_test[0]

        hter_test = result_test[3]
        acer_test = result_test[-1]

        if acer_test < acer_best and epoch > 15:
            acer_best = acer_test
            save_path = os.path.join(args.model_root, args.name + '_acer_best_' + '.pth')
            torch.save(student_model.state_dict(), save_path, )

        if hter_test < hter_best and epoch > 15:
            hter_best = hter_test
            save_path = os.path.join(args.model_root, args.name + '_hter_best_' + '.pth')
            torch.save(student_model.state_dict(), save_path, )

        if accuracy_test > accuracy_best and epoch > 15:
            accuracy_best = accuracy_test
            save_path = os.path.join(args.model_root, args.name + '.pth')
            torch.save(student_model.state_dict(), save_path, )
        log_list.append(train_loss / len(train_loader))
        log_list.append(accuracy_test)
        log_list.append(accuracy_best)
        print(cls_loss_sum / len(train_loader), kd_logits_loss_sum / (len(train_loader)),
              kd_feature_loss_sum / (len(train_loader)))

        print(
            "Epoch {}, loss={:.5f}, accuracy_test={:.5f},  accuracy_best={:.5f}".format(epoch,
                                                                                        train_loss / len(
                                                                                            train_loader),
                                                                                        accuracy_test, accuracy_best))
        train_loss = 0
        cls_loss_sum = 0
        kd_feature_loss_sum = 0
        kd_logits_loss_sum = 0

        if args.lr_decrease:
            if args.lr_warmup:
                scheduler_warmup.step(epoch=epoch)
            else:
                cos_scheduler.step(epoch=epoch)
        if epoch < 20:
            print(epoch, optimizer.param_groups[0]['lr'])

        # save model and para
        if epoch % save_interval == 0:
            train_state = {
                "Epoch": epoch,
                "model_state": student_model.state_dict(),
                "optim_state": optimizer.state_dict(),
                "args": args
            }
            models_dir = args.model_root + '/' + args.name + '.pt'
            torch.save(train_state, models_dir)

        #  save log
        with open(log_dir, 'a+', newline='') as f:
            # 训练结果
            my_writer = csv.writer(f)
            my_writer.writerow(log_list)
            log_list = []
        epoch = epoch + 1
    # train_duration_sec = int(time.time() - start)
    # print("training is end", train_duration_sec)


def train_knowledge_distill_patch_feature_cefa(net_dict, cost_dict, optimizer, train_loader, test_loader, args):
    '''

    :param model:
    :param cost:
    :param optimizer:
    :param train_loader:
    :param test_loader:
    :param args:
    :return:
    '''
    from loss.kd.pkt import PKTCosSim
    from loss.kd.sp import SP
    from loss.kd.at import AT
    print(args)
    if not os.path.exists(args.model_root):
        os.makedirs(args.model_root)
    if not os.path.exists(args.log_root):
        os.makedirs(args.log_root)

    models_dir = args.model_root + '/' + args.name + '.pt'
    log_dir = args.log_root + '/' + args.name + '.csv'

    # save args
    with open(log_dir, 'a+', newline='') as f:
        my_writer = csv.writer(f)
        args_dict = vars(args)
        for key, value in args_dict.items():
            my_writer.writerow([key, value])
        f.close()

    student_model = net_dict['snet']
    teacher_model = net_dict['tnet']

    criterionCls = cost_dict['criterionCls']
    criterionKD = cost_dict['criterionKD']

    if torch.cuda.is_available():
        mmd_loss = MMD_loss().cuda()
    else:
        mmd_loss = MMD_loss()

    if torch.cuda.is_available():
        pkt_loss = PKTCosSim().cuda()
    else:
        pkt_loss = PKTCosSim()

    if torch.cuda.is_available():
        sp_loss = SP().cuda()
    else:
        sp_loss = SP()

    if torch.cuda.is_available():
        bce_loss = nn.BCELoss().cuda()
    else:
        bce_loss = nn.BCELoss()

    if torch.cuda.is_available():
        mse_loss = nn.MSELoss().cuda()
    else:
        mse_loss = nn.MSELoss()

    if torch.cuda.is_available():
        at_loss = AT(p=2).cuda()
    else:
        at_loss = AT(p=2)

    #  learning rate decay
    if args.lr_decrease == 'cos':
        print("lrcos is using")
        cos_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.train_epoch + 20, eta_min=0)

        if args.lr_warmup:
            scheduler_warmup = GradualWarmupScheduler(args, optimizer, multiplier=1,
                                                      after_scheduler=cos_scheduler)
    elif args.lr_decrease == 'multi_step':
        print("multi_step is using")
        cos_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[np.int(args.train_epoch * 1 / 6),
                                                                              np.int(args.train_epoch * 2 / 6),
                                                                              np.int(args.train_epoch * 3 / 6)])
        if args.lr_warmup:
            scheduler_warmup = GradualWarmupScheduler(args, optimizer, multiplier=1,
                                                      after_scheduler=cos_scheduler)

    # Training initialization
    epoch_num = args.train_epoch
    log_interval = args.log_interval
    save_interval = args.save_interval
    batch_num = 0
    train_loss = 0
    cls_loss_sum = 0
    kd_logits_loss_sum = 0
    kd_feature_loss_sum = 0
    epoch = 0
    accuracy_best = 0
    acer_best = 1
    hter_best = 1
    log_list = []  # log need to save

    if args.retrain:
        if not os.path.exists(models_dir):
            print("no trained model")
        else:
            state_read = torch.load(models_dir)
            student_model.load_state_dict(state_read['model_state'])
            optimizer.load_state_dict(state_read['optim_state'])
            epoch = state_read['Epoch']
            print("retaining")

    # Train
    while epoch < epoch_num:
        import datetime
        start = datetime.datetime.now()

        for batch_idx, multi_sample in enumerate(
                tqdm(train_loader, desc="Epoch {}/{}".format(epoch, epoch_num))):

            if epoch == 0:
                continue

            data_read_time = (datetime.datetime.now() - start)
            # print("data_read_time:", data_read_time.total_seconds())
            start = datetime.datetime.now()
            batch_num += 1

            img_rgb, img_ir, img_depth, target = multi_sample['image_x'], multi_sample['image_ir'], \
                multi_sample['image_depth'], multi_sample[
                'binary_label']

            if torch.cuda.is_available():
                img_rgb = img_rgb.cuda()
                img_ir = img_ir.cuda()
                img_depth = img_depth.cuda()
                target = target.cuda()
            label = target

            optimizer.zero_grad()

            teacher_whole_out, teacher_patch_out, teacher_layer3, teacher_layer4 = teacher_model(img_rgb, img_ir,
                                                                                                 img_depth)

            student_whole_out, student_patch_out, student_layer3, student_layer4 = student_model(img_rgb)

            kd_feature_loss_1 = at_loss(student_layer3, teacher_layer3)
            kd_feature_loss_2 = at_loss(student_layer4, teacher_layer4)
            kd_feature_loss = kd_feature_loss_1 + kd_feature_loss_2

            # 分类损失
            if args.student_data == 'multi_rgb':
                cls_loss = criterionCls(student_whole_out, target)
            else:
                cls_loss = criterionCls(student_whole_out, label)

            cls_loss = cls_loss.cuda()

            loss = cls_loss + kd_feature_loss

            train_loss += loss.item()
            cls_loss_sum += cls_loss.item()
            kd_feature_loss_sum += kd_feature_loss.item()
            loss.backward()
            optimizer.step()
            # if batch_idx % log_interval == 0:  # 准备打印相关信息，args.log_interval是最开头设置的好了的参数
            #     print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            #         epoch, batch_idx * len(data), len(train_loader.dataset),
            #                100. * batch_idx / len(train_loader), loss.item()))

        # testing
        result_test = calc_accuracy_kd_patch_feature(model=student_model, args=args, loader=test_loader, hter=True)
        print(result_test)
        accuracy_test = result_test[0]

        hter_test = result_test[3]
        acer_test = result_test[-1]

        if acer_test < acer_best and epoch > 15:
            acer_best = acer_test
            save_path = os.path.join(args.model_root, args.name + '_acer_best_' + '.pth')
            torch.save(student_model.state_dict(), save_path, )

        if hter_test < hter_best and epoch > 15:
            hter_best = hter_test
            save_path = os.path.join(args.model_root, args.name + '_hter_best_' + '.pth')
            torch.save(student_model.state_dict(), save_path, )

        if accuracy_test > accuracy_best and epoch > 15:
            accuracy_best = accuracy_test
            save_path = os.path.join(args.model_root, args.name + '.pth')
            torch.save(student_model.state_dict(), save_path, )
        log_list.append(train_loss / len(train_loader))
        log_list.append(accuracy_test)
        log_list.append(accuracy_best)
        print(cls_loss_sum / len(train_loader), kd_logits_loss_sum / (len(train_loader)),
              kd_feature_loss_sum / (len(train_loader)))

        print(
            "Epoch {}, loss={:.5f}, accuracy_test={:.5f},  accuracy_best={:.5f}".format(epoch,
                                                                                        train_loss / len(
                                                                                            train_loader),
                                                                                        accuracy_test, accuracy_best))
        train_loss = 0
        cls_loss_sum = 0
        kd_feature_loss_sum = 0
        kd_logits_loss_sum = 0

        if args.lr_decrease:
            if args.lr_warmup:
                scheduler_warmup.step(epoch=epoch)
            else:
                cos_scheduler.step(epoch=epoch)
        if epoch < 20:
            print(epoch, optimizer.param_groups[0]['lr'])

        # save model and para
        if epoch % save_interval == 0:
            train_state = {
                "Epoch": epoch,
                "model_state": student_model.state_dict(),
                "optim_state": optimizer.state_dict(),
                "args": args
            }
            models_dir = args.model_root + '/' + args.name + '.pt'
            torch.save(train_state, models_dir)

        #  save log
        with open(log_dir, 'a+', newline='') as f:
            # 训练结果
            my_writer = csv.writer(f)
            my_writer.writerow(log_list)
            log_list = []
        epoch = epoch + 1
    train_duration_sec = int(time.time() - start)
    print("training is end", train_duration_sec)
