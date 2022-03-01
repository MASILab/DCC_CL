"""ResNet in PyTorch.
ImageNet-Style ResNet
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
Adapted from: https://github.com/bearpaw/pytorch-classification
"""

import torch
import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
from torch.nn import functional as F

class ASPP(nn.Module):

    def __init__(self, C, depth, num_classes, conv=nn.Conv2d, norm=nn.BatchNorm2d, momentum=0.0003, mult=1):
        super(ASPP, self).__init__()
        self._C = C
        self._depth = depth
        self._num_classes = num_classes

        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.relu = nn.ReLU(inplace=True)
        self.aspp1 = conv(C, depth, kernel_size=1, stride=1, bias=False)
        self.aspp2 = conv(C, depth, kernel_size=3, stride=1,
                               dilation=int(6*mult), padding=int(6*mult),
                               bias=False)
        self.aspp3 = conv(C, depth, kernel_size=3, stride=1,
                               dilation=int(12*mult), padding=int(12*mult),
                               bias=False)
        self.aspp4 = conv(C, depth, kernel_size=3, stride=1,
                               dilation=int(18*mult), padding=int(18*mult),
                               bias=False)
        self.aspp5 = conv(C, depth, kernel_size=1, stride=1, bias=False)
        self.aspp1_bn = norm(depth, momentum)
        self.aspp2_bn = norm(depth, momentum)
        self.aspp3_bn = norm(depth, momentum)
        self.aspp4_bn = norm(depth, momentum)
        self.aspp5_bn = norm(depth, momentum)
        self.conv2 = conv(depth * 5, depth, kernel_size=1, stride=1,
                               bias=False)
        self.bn2 = norm(depth, momentum)
        self.conv3 = nn.Conv2d(depth, num_classes, kernel_size=1, stride=1)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x1 = self.aspp1(x)
        x1 = self.aspp1_bn(x1)
        x1 = self.relu(x1)
        x2 = self.aspp2(x)
        x2 = self.aspp2_bn(x2)
        x2 = self.relu(x2)
        x3 = self.aspp3(x)
        x3 = self.aspp3_bn(x3)
        x3 = self.relu(x3)
        x4 = self.aspp4(x)
        x4 = self.aspp4_bn(x4)
        x4 = self.relu(x4)
        x5 = self.global_pooling(x)
        x5 = self.aspp5(x5)
        x5 = self.aspp5_bn(x5)
        x5 = self.relu(x5)
        x5 = nn.Upsample((x.shape[2], x.shape[3]), mode='bilinear',
                         align_corners=True)(x5)
        x = torch.cat((x1, x2, x3, x4, x5), 1)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x)
        # x = self.tanh(x)

        return x


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, is_last=False):
        super(BasicBlock, self).__init__()
        self.is_last = is_last
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        preact = out
        out = F.relu(out)
        if self.is_last:
            return out, preact
        else:
            return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, is_last=False):
        super(Bottleneck, self).__init__()
        self.is_last = is_last
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        preact = out
        out = F.relu(out)
        if self.is_last:
            return out, preact
        else:
            return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, in_channel=6, zero_init_residual=False):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(in_channel, 64, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves
        # like an identity. This improves the model by 0.2~0.3% according to:
        # https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for i in range(num_blocks):
            stride = strides[i]
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, layer=100):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        # print(out.size())
        # out = self.avgpool(out)
        # out = torch.flatten(out, 1)
        return out


def resnet18(**kwargs):
    return ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)


def resnet34(**kwargs):
    return ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)


def resnet50(**kwargs):
    return ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)


def resnet101(**kwargs):
    return ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)


model_dict = {
    'resnet18': [resnet18, 512],
    'resnet34': [resnet34, 512],
    'resnet50': [resnet50, 2048],
    'resnet101': [resnet101, 2048],
}

class VGGBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels, act_func=nn.ReLU(inplace=True)):
        super(VGGBlock, self).__init__()
        self.act_func = act_func
        self.conv1 = nn.Conv2d(in_channels, middle_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(middle_channels)
        self.conv2 = nn.Conv2d(middle_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x_1):
        out = self.conv1(x_1)
        out = self.bn1(out)
        out = self.act_func(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.act_func(out)

        return out

class NestedUNet(nn.Module):
    def __init__(self, input_channels, out_channels):
        super().__init__()

        # self.args = args
        self.deepsupervision = True
        self.input_channels = input_channels
        self.out_channels = out_channels

        nb_filter = [32, 64, 128, 256, 512]

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv0_0 = VGGBlock(self.input_channels, nb_filter[0], nb_filter[0])
        self.conv1_0 = VGGBlock(nb_filter[0], nb_filter[1], nb_filter[1])
        self.conv2_0 = VGGBlock(nb_filter[1], nb_filter[2], nb_filter[2])
        self.conv3_0 = VGGBlock(nb_filter[2], nb_filter[3], nb_filter[3])
        self.conv4_0 = VGGBlock(nb_filter[3], nb_filter[4], nb_filter[4])

        self.conv0_1 = VGGBlock(nb_filter[0]+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_1 = VGGBlock(nb_filter[1]+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_1 = VGGBlock(nb_filter[2]+nb_filter[3], nb_filter[2], nb_filter[2])
        self.conv3_1 = VGGBlock(nb_filter[3]+nb_filter[4], nb_filter[3], nb_filter[3])

        self.conv0_2 = VGGBlock(nb_filter[0]*2+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_2 = VGGBlock(nb_filter[1]*2+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_2 = VGGBlock(nb_filter[2]*2+nb_filter[3], nb_filter[2], nb_filter[2])

        self.conv0_3 = VGGBlock(nb_filter[0]*3+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_3 = VGGBlock(nb_filter[1]*3+nb_filter[2], nb_filter[1], nb_filter[1])

        self.conv0_4 = VGGBlock(nb_filter[0]*4+nb_filter[1], nb_filter[0], nb_filter[0])

        if self.deepsupervision == True:
            self.final1 = nn.Conv2d(nb_filter[0], self.out_channels, kernel_size=1)
            self.final2 = nn.Conv2d(nb_filter[0], self.out_channels, kernel_size=1)
            self.final3 = nn.Conv2d(nb_filter[0], self.out_channels, kernel_size=1)
            self.final4 = nn.Conv2d(nb_filter[0], self.out_channels, kernel_size=1)
        else:
            self.final = nn.Conv2d(nb_filter[0], 1, kernel_size=1)


    def forward(self, x):
        x0_0 = self.conv0_0(x)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))

        x2_0 = self.conv2_0(self.pool(x1_0))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))

        x3_0 = self.conv3_0(self.pool(x2_0))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1))

        x4_0 = self.conv4_0(self.pool(x3_0))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], 1))

        return [x0_1, x0_2, x0_3, x0_4]




class LinearBatchNorm(nn.Module):
    """Implements BatchNorm1d by BatchNorm2d, for SyncBN purpose"""
    def __init__(self, dim, affine=True):
        super(LinearBatchNorm, self).__init__()
        self.dim = dim
        self.bn = nn.BatchNorm2d(dim, affine=affine)

    def forward(self, x):
        x = x.view(-1, self.dim, 1, 1)
        x = self.bn(x)
        x = x.view(-1, self.dim)
        return x


class SupConResNet(nn.Module):
    """backbone + projection head"""
    def __init__(self, name='resnet50', head='mlp', feat_dim=256):
        super(SupConResNet, self).__init__()
        model_fun, dim_in = model_dict[name]
        self.encoder = model_fun()
        if head == 'linear':
            self.head = nn.Linear(dim_in, feat_dim)
        elif head == 'mlp':
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.head = nn.Sequential(
                nn.Linear(dim_in, dim_in),
                nn.ReLU(inplace=True),
                nn.Linear(dim_in, feat_dim)
            )
        else:
            raise NotImplementedError(
                'head not supported: {}'.format(head))

    def forward(self, x):
        feat_int = self.encoder(x)
        feat = self.avgpool(feat_int)
        # print(feat.size())
        feat = torch.flatten(feat, 1)
        # print(feat.size())
        con_feat = F.normalize(self.head(feat), dim=1)
        # print(feat.size())
        return con_feat, feat_int

class Conv2d(nn.Conv2d):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super(Conv2d, self).__init__(in_channels, out_channels, kernel_size, stride,
                 padding, dilation, groups, bias)

    def forward(self, x):
        # return super(Conv2d, self).forward(x)
        weight = self.weight
        weight_mean = weight.mean(dim=1, keepdim=True).mean(dim=2,
                                  keepdim=True).mean(dim=3, keepdim=True)
        weight = weight - weight_mean
        std = weight.view(weight.size(0), -1).std(dim=1).view(-1, 1, 1, 1) + 1e-5
        weight = weight / std.expand_as(weight)
        return F.conv2d(x, weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)
    
class ContrastSeg(nn.Module):
    def __init__(self, name='resnet50', head='mlp', feat_dim=256, num_classes=2, weight_std=False):
        super(ContrastSeg, self).__init__()
        model_fun, dim_in = model_dict[name]
        self.encoder = model_fun()
        if head == 'linear':
            self.head = nn.Linear(dim_in, feat_dim)
        elif head == 'mlp':
            # self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.head = nn.Sequential(
                nn.Conv2d(dim_in, dim_in, kernel_size=1),
                nn.BatchNorm2d(dim_in),
                nn.ReLU(),
                nn.Conv2d(dim_in, feat_dim, kernel_size=1)
            )
        else:
            raise NotImplementedError(
                'head not supported: {}'.format(head))

        expansion = 4
        self.norm = lambda planes, momentum=0.05: nn.BatchNorm2d(planes, momentum=momentum)
        self.conv = Conv2d if weight_std else nn.Conv2d
        self.aspp = ASPP(512 * expansion, 512, num_classes, conv=self.conv, norm=self.norm)

    def forward(self, x):
        feat_int = self.encoder(x)
        # feat = self.avgpool(feat_int)
        # # print(feat.size())
        # feat = torch.flatten(feat, 1)
        # print(feat.size())
        con_feat = F.normalize(self.head(feat_int), dim=1)

        size = (int(128),int(128))
        x = self.aspp(feat_int)
        x = nn.Upsample(size, mode='bilinear', align_corners=True)(x)
        # print(feat.size())
        return con_feat, x

class DenseResNet(nn.Module):
    """backbone + projection head"""
    def __init__(self, name='resnet50', head='dense', feat_dim=256, num_classes=2):
        super(DenseResNet, self).__init__()
        model_fun, dim_in = model_dict[name]
        self.encoder = model_fun()
        if head == 'linear':
            self.head = nn.Linear(dim_in, feat_dim)
        elif head == 'dense':
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.avgpool_2 = nn.AdaptiveAvgPool2d((1, 1))
            self.dense_head = nn.Sequential(
                    nn.Conv2d(dim_in, dim_in, 1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(dim_in, feat_dim, 1)
                    )
            self.mlp = nn.Sequential(
                nn.Linear(dim_in, dim_in),
                nn.ReLU(inplace=True),
                nn.Linear(dim_in, feat_dim)
            )
        else:
            raise NotImplementedError(
                'head not supported: {}'.format(head))

    def forward(self, x):
        feat = self.encoder(x)
        avgpooled_x = self.avgpool(feat)
        avgpooled_x = torch.flatten(avgpooled_x, 1)
        avgpooled_x = F.normalize(self.mlp(avgpooled_x), dim=1)
        
        a = self.dense_head(feat)
        # print(a.size())
        dense = F.normalize(self.dense_head(feat), dim=1)
        avgpooled_x2 = self.avgpool_2(dense)
        dense = dense.view(dense.size(0), dense.size(1), -1)
        avgpooled_x2 = torch.flatten(avgpooled_x2, 1)
        
        return [avgpooled_x, dense, avgpooled_x2, feat]
          

class SupConUNet(nn.Module):
    """backbone + projection head"""
    def __init__(self, name='unet_nested', head='linear', feat_dim=256, output_channel=2):
        super(SupConUNet, self).__init__()
        self.encoder = NestedUNet(input_channels=6, out_channels=output_channel)
        if head == 'linear':
            self.head = nn.Linear(128*64*64, feat_dim)
        elif head == 'mlp':
            self.head = nn.Sequential(
                nn.Linear(128*64*64, 128*64*64),
                nn.ReLU(inplace=True),
                nn.Linear(128*64*64, feat_dim)
            )
        else:
            raise NotImplementedError(
                'head not supported: {}'.format(head))

    def forward(self, x):
        feat = self.encoder(x)
        feat_f = []
        for f in feat:
            f = torch.flatten(f, 1)
            f = F.normalize(self.head(f), dim=1)
            feat_f.append(f)
        return feat_f


class SupCEResNet(nn.Module):
    """encoder + classifier"""
    def __init__(self, name='resnet50', num_classes=24):
        super(SupCEResNet, self).__init__()
        model_fun, dim_in = model_dict[name]
        self.encoder = model_fun()
        self.fc = nn.Linear(dim_in, num_classes)
        self.sigm = nn.Sigmoid()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x_e = self.encoder(x)
        x_e = self.avgpool(x_e)
        out = torch.flatten(x_e, 1)
        feat = self.fc(out)
        feat = self.sigm(feat)
        return feat


class LinearClassifier(nn.Module):
    """Linear classifier"""
    def __init__(self, name='resnet50', num_classes=2):
        super(LinearClassifier, self).__init__()
        _, feat_dim = model_dict[name]
        self.fc = nn.Linear(feat_dim, num_classes)
        
    def forward(self, features):
        return self.fc(features)
    
class recon_decoder(nn.Module):
    "Image Reconstruction"
    def __init__(self, name='resnet50', num_classes=3, weight_std=False, beta=False):
        super(recon_decoder, self).__init__()
        if name == 'resnet18':
            expansion = 1
        elif name == 'resnet50':
            expansion = 4
        elif name == 'resnet101':
            expansion = 4
        _, feat_dim = model_dict[name]
        self.norm = lambda planes, momentum=0.05: nn.BatchNorm2d(planes, momentum=momentum)
        self.conv = Conv2d if weight_std else nn.Conv2d
        self.aspp = ASPP(512 * expansion, 512, num_classes, conv=self.conv, norm=self.norm)
    
    def forward(self, features):
        size = (int(128),int(128))
        # features = torch.unsqueeze(features, 2)
        # features = torch.unsqueeze(features, 3)
        x = self.aspp(features)
        x = nn.Upsample(size, mode='bilinear', align_corners=True)(x)
        
        return x
    
class SegmentationModel(nn.Module):
    """Multi-Organ Segmentation"""
    def __init__(self, name='resnet101', vit=None, num_classes=2, weight_std=False, beta=False):
        super(SegmentationModel, self).__init__()
        if name == 'resnet18':
            expansion = 1
        elif name == 'resnet50':
            expansion = 4
        elif name == 'resnet101':
            expansion = 4
        elif name == 'vit':
            self.vit = vit
            expansion = 4
        
        if name == 'vit':
            feat_dim = 1024
        else:
            _, feat_dim = model_dict[name]
        self.norm = lambda planes, momentum=0.05: nn.BatchNorm2d(planes, momentum=momentum)
        self.conv = Conv2d if weight_std else nn.Conv2d
        self.aspp = ASPP(512 * expansion, 512, num_classes, conv=self.conv, norm=self.norm)
        
    def forward(self, features):
        size = (int(128),int(128))
        # features = torch.unsqueeze(features, 2)
        # features = torch.unsqueeze(features, 3)
        x = self.aspp(features)
        x = nn.Upsample(size, mode='bilinear', align_corners=True)(x)
        
        return x

class SegmentationModel_unet(nn.Module):
    """Multi-Organ Segmentation"""
    def __init__(self, name='unet_nested', out_channels=2):
        super(SegmentationModel_unet, self).__init__()
        nb_filter = [32, 64, 128, 256, 512]
        self.final1 = nn.Conv2d(nb_filter[0], out_channels, kernel_size=1)
        self.final2 = nn.Conv2d(nb_filter[0], out_channels, kernel_size=1)
        self.final3 = nn.Conv2d(nb_filter[0], out_channels, kernel_size=1)
        self.final4 = nn.Conv2d(nb_filter[0], out_channels, kernel_size=1)
        
    def forward(self, features):
        output1 = self.final1(features[0])
        output2 = self.final2(features[1])
        output3 = self.final3(features[2])
        output4 = self.final4(features[3])
        return [output1, output2, output3, output4]
    
    


