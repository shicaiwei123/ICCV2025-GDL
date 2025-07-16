import torch
import torch.nn as nn
import torch.nn.functional as F
from .backbone import resnet18
from .fusion_modules import SumFusion, ConcatFusion, FiLM, GatedFusion, ConcatFusion_Swin,ConcatFusion_DGL,GatedFusion_DGL,SumFusion_DGL,FiLM_DGL,ConcatFusion_DGL_unimodal
import numpy as np
from models.swin_transformer import SwinTransformer


class AVClassifier_DGL(nn.Module):
    def __init__(self, args):
        super(AVClassifier_DGL, self).__init__()

        fusion = args.fusion_method
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

        if fusion == 'sum':
            self.fusion_module = SumFusion_DGL(output_dim=n_classes)
        elif fusion == 'concat':
            if args.dataset == 'kinect400':
                self.fusion_module = ConcatFusion_DGL(output_dim=n_classes,input_dim=1024)
            else:
                self.fusion_module = ConcatFusion_DGL(output_dim=n_classes)
        elif fusion == 'film':
            self.fusion_module = FiLM_DGL(output_dim=n_classes, x_film=True)
        elif fusion == 'gated':
            self.fusion_module = GatedFusion_DGL(output_dim=n_classes, x_gate=True)
        else:
            raise NotImplementedError('Incorrect fusion method: {}!'.format(fusion))

        if args.modality == 'full':
            self.audio_net = resnet18(modality='audio', args=args)
            self.visual_net = resnet18(modality='visual', args=args)

        if args.modality == 'visual':
            if args.dataset == 'kinect400':
                self.visual_net = resnet18(modality='visual', args=args)
                self.visual_classifier = nn.Linear(512, n_classes)
            else:
                self.visual_net = resnet18(modality='visual', args=args)
                self.visual_classifier = nn.Linear(512, n_classes)
        if args.modality == 'audio':
            if args.dataset == 'kinect400':
                self.audio_net = resnet18(modality='audio', args=args)
                self.audio_classifier = nn.Linear(512, n_classes)
            else:
                self.audio_net = resnet18(modality='audio', args=args)
                self.audio_classifier = nn.Linear(512, n_classes)
        self.modality = args.modality
        self.args = args



    def forward(self, audio, visual):

        if self.modality == 'full':

            
            a = self.audio_net(audio)  # only feature
            v = self.visual_net(visual)

            (_, C, H, W) = v.size()
            B = a.size()[0]
            v = v.view(B, -1, C, H, W)
            v = v.permute(0, 2, 1, 3, 4)

            a = F.adaptive_avg_pool2d(a, 1)
            v = F.adaptive_avg_pool3d(v, 1)

            a = torch.flatten(a, 1)
            v = torch.flatten(v, 1)

            a_out, v_out, out = self.fusion_module(a, v)  # av 是原来的，out是融合结果

            return  out,a_out,v_out
        
        elif self.modality == 'visual':

            v = self.visual_net(visual)


            (_, C, H, W) = v.size()
            B = self.args.batch_size
            v = v.view(B, -1, C, H, W)

            v = v.permute(0, 2, 1, 3, 4)

            v = F.adaptive_avg_pool3d(v, 1)

            v = torch.flatten(v, 1)

            out = self.visual_classifier(v)

            a = torch.zeros_like(v)


            return out, out, out

        elif self.modality == 'audio':

            a = self.audio_net(audio)  # only feature
            a_feature = a

            a = F.adaptive_avg_pool2d(a, 1)

            a = torch.flatten(a, 1)

            out = self.audio_classifier(a)
            v = torch.zeros_like(a)

            return out, out,out
        else:
            return 0, 0, 0

