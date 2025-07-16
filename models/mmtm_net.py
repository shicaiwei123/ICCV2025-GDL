import torch.nn as nn
import torch
import torchvision.models as tm

from models.resnet18_se import resnet18_se
import torchvision.transforms.functional as F


def init_weights(m):
    print(m)
    if type(m) == nn.Linear:
        print(m.weight)
    else:
        print('error')


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, args, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.args = args
        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=2, padding=1,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        # if args.modal == 'multi':
        #     self.inplanes = args.inplace_new
        #     self.layer3_new = self._make_layer(block, 256, layers[2], stride=2,
        #                                        dilate=replace_stride_with_dilation[1])
        # else:
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        self.dropout = nn.Dropout(p=0.5)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        layer3 = self.layer3(x)
        layer4 = self.layer4(layer3)

        x = self.avgpool(layer4)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x, layer3, layer4


def resnet18_se(args, pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], args=args, num_classes=args.class_num, **kwargs)
    if not pretrained:
        return model
    else:
        resnet_18_pretrain = tm.resnet18(pretrained=True)
        pretrain_para_dict = resnet_18_pretrain.state_dict()
        init_para_dict = model.state_dict()
        try:
            for k, v in pretrain_para_dict.items():
                if k in init_para_dict:
                    init_para_dict[k] = pretrain_para_dict[k]
            model.load_state_dict(init_para_dict)
        except Exception as e:
            print(e)

        return model


class MMTM(nn.Module):
    def __init__(self, dim_visual, dim_skeleton, ratio):
        super(MMTM, self).__init__()
        dim = dim_visual + dim_skeleton
        dim_out = int(2 * dim / ratio)
        self.fc_squeeze = nn.Linear(dim, dim_out)

        self.fc_visual = nn.Linear(dim_out, dim_visual)
        self.fc_skeleton = nn.Linear(dim_out, dim_skeleton)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        # initialize
        with torch.no_grad():
            self.fc_squeeze.apply(init_weights)
            self.fc_visual.apply(init_weights)
            self.fc_skeleton.apply(init_weights)

    def forward(self, visual, skeleton):
        squeeze_array = []
        for tensor in [visual, skeleton]:
            tview = tensor.view(tensor.shape[:2] + (-1,))
            squeeze_array.append(torch.mean(tview, dim=-1))
        squeeze = torch.cat(squeeze_array, 1)

        excitation = self.fc_squeeze(squeeze)
        excitation = self.relu(excitation)

        vis_out = self.fc_visual(excitation)
        sk_out = self.fc_skeleton(excitation)

        vis_out = self.sigmoid(vis_out)
        sk_out = self.sigmoid(sk_out)

        dim_diff = len(visual.shape) - len(vis_out.shape)
        vis_out = vis_out.view(vis_out.shape + (1,) * dim_diff)

        dim_diff = len(skeleton.shape) - len(sk_out.shape)
        sk_out = sk_out.view(sk_out.shape + (1,) * dim_diff)

        return visual * vis_out, skeleton * sk_out


class estimate_mean_std(nn.Module):
    def __init__(self, input_channel, output_channel):
        input = input_channel
        output = output_channel
        super().__init__()

        self.mu_dul_backbone = nn.Sequential(
            nn.Conv2d(input, output, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(output),
        )
        self.logvar_dul_backbone = nn.Sequential(
            nn.Conv2d(input, output, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(output),
        )

    def forward(self, x, scale=1.0):

        mu_dul = self.mu_dul_backbone(x)
        logvar_dul = self.logvar_dul_backbone(x)
        std_dul = (logvar_dul * 0.5).exp()

        # std_dul = torch.mean(std_dul, dim=(2, 3))
        # std_dul = torch.unsqueeze(std_dul, dim=2)
        # std_dul = torch.unsqueeze(std_dul, dim=3)
        # print(std_dul)

        epsilon = torch.randn_like(mu_dul)

        # if Epoch < 5:
        #     std_dul =std_dul* torch.zeros_like(mu_dul).cuda()

        if self.training:
            features = mu_dul + epsilon * std_dul * scale
        else:
            features = mu_dul

        return features, mu_dul, std_dul


class MMTM_Net(nn.Module):
    def __init__(self, args):
        super().__init__()

        model_resnet18_se_1 = resnet18_se(args, pretrained=False)
        model_resnet18_se_3 = resnet18_se(args, pretrained=False)
        self.args = args

        self.special_bone_rgb = nn.Sequential(model_resnet18_se_1.conv1,
                                              model_resnet18_se_1.bn1,
                                              model_resnet18_se_1.relu,
                                              model_resnet18_se_1.maxpool,
                                              model_resnet18_se_1.layer1,
                                              model_resnet18_se_1.layer2)

        self.special_bone_depth = nn.Sequential(model_resnet18_se_3.conv1,
                                                model_resnet18_se_3.bn1,
                                                model_resnet18_se_3.relu,
                                                model_resnet18_se_3.maxpool,
                                                model_resnet18_se_3.layer1,
                                                model_resnet18_se_3.layer2)

        self.rgb_0 = model_resnet18_se_1.layer3
        self.rgb_1 = model_resnet18_se_1.layer4

        self.depth_0 = model_resnet18_se_1.layer3
        self.depth_1 = model_resnet18_se_1.layer4

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc = nn.Linear(1024, args.class_num)

        self.mmtm0 = MMTM(128, 128, 2)
        self.mmtm1 = MMTM(256, 256, 2)
        self.mmtm2 = MMTM(512, 512, 2)


    def fusion(self,x_rgb,x_depth):


        x_rgb, x_depth = self.mmtm0(x_rgb, x_depth)

        x_rgb = self.rgb_0(x_rgb)
        x_depth = self.depth_0(x_depth)

        x_rgb, x_depth = self.mmtm1(x_rgb, x_depth)

        x_rgb = self.rgb_1(x_rgb)
        x_depth = self.depth_1(x_depth)

        x_rgb, x_depth = self.mmtm2(x_rgb, x_depth)

        x_rgb = self.avgpool(x_rgb)
        x_depth = self.avgpool(x_depth)

        x_rgb = x_rgb.view((x_rgb.shape[0]), -1)
        x_depth = x_depth.view(x_depth.shape[0], -1)

        x = torch.cat((x_rgb, x_depth), dim=1)
        return x


    def forward(self, img_rgb, img_depth):
        if self.args.dataset == 'AVE' or self.args.dataset == 'CREMAD' or self.args.dataset == 'KineticSound':
            img_rgb = torch.unsqueeze(img_rgb, dim=1)
            img_rgb = torch.repeat_interleave(img_rgb, dim=1, repeats=3)
            # print(img_rgb.shape)
            img_rgb = F.resize(img_rgb, (224, 224))

            img_depth = torch.squeeze(img_depth, dim=2)


        x_rgb = self.special_bone_rgb(img_rgb)
        x_depth = self.special_bone_depth(img_depth)

        x_rgb_detach=x_rgb
        x_depth_detach=x_depth

        # x_depth_detach=torch.zeros_like(x_depth)


        x=self.fusion(x_rgb_detach,x_depth_detach)
        out = self.fc(x)

        out_rgb=self.fusion(x_rgb,torch.zeros_like(x_depth))
        out_depth=self.fusion(torch.zeros_like(x_rgb),x_depth)

        out_rgb=self.fc(out_rgb)
        out_depth=self.fc(out_depth)

        return out, out_rgb, out_depth