import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import  math
# from .utils import load_state_dict_from_url
import pdb
import torch.nn.functional as torchf
from utils.misc import SoftSigmoid
from torch.utils.model_zoo import load_url as load_state_dict_from_url

# device = torch.device("cuda:1" if torch.cuda.is_available() > 0 else "cpu")
eps = torch.finfo().eps

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
}


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)



class P1ECALayer(nn.Module):
    """Constructs a ECA module.
    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """

    def __init__(self, channel):
        super(P1ECALayer, self).__init__()
        self.avg_pool1 = nn.AdaptiveAvgPool2d(1)
        self.avg_pool2 = nn.AdaptiveAvgPool2d(2)
        self.avg_pool3 = nn.AdaptiveAvgPool2d(4)
        self.conv = nn.Conv1d(1, 1, kernel_size=3, padding=1, bias=False)
        self.Squeeze2 = nn.Conv2d(channel, channel, kernel_size=2, groups=channel, bias=False)
        self.Squeeze4 = nn.Conv2d(channel, channel, kernel_size=4, groups=channel, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # feature descriptor on the global spatial information
        b, c, _, _ = x.size()
        y1 = self.avg_pool1(x)
        y2 = self.avg_pool2(x)
        y2 = self.Squeeze2(y2)
        y3 = self.avg_pool3(x)
        y3 = self.Squeeze4(y3)
        # Two different branches of ECA module
        y1 = self.conv(y1.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y2 = self.conv(y2.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y3 = self.conv(y3.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        # Multi-scale information fusion
        y1 = self.sigmoid(y1)
        y2 = self.sigmoid(y2)
        y3 = self.sigmoid(y3)
        y = y1 + y2 + y3
        y = self.sigmoid(y)
        return x * y.expand_as(x)

class P2ECALayer(nn.Module):
    """Constructs a ECA module.
    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """

    def __init__(self, channel, k_size=3):
        super(P2ECALayer, self).__init__()
        self.avg_pool1 = nn.AdaptiveAvgPool2d(1)
        self.avg_pool2 = nn.AdaptiveAvgPool2d(2)
        self.avg_pool3 = nn.AdaptiveAvgPool2d(3)
        self.conv = nn.Conv1d(1, 1, kernel_size=3, padding=1, bias=False)
        self.conv4 = nn.Conv2d(2, 1, kernel_size=(3,2), padding=(1,0), bias=False)
        self.conv9 = nn.Conv2d(3, 1, kernel_size=(3,3), padding=(1,0),bias=False)
        self.sigmoid = nn.Sigmoid()
        self.relu=nn.ReLU(inplace=True)

    def forward(self, x):
        # feature descriptor on the global spatial information
        y1 = self.avg_pool1(x)
        y2 = self.avg_pool2(x).permute(0, 2, 1, 3)
        y3 = self.avg_pool3(x).permute(0, 2, 1, 3)
        # Two different branches of ECA module
        y1 = self.conv(y1.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y2 = self.conv4(y2).permute(0, 2, 1, 3)
        y3 = self.conv9(y3).permute(0, 2, 1, 3)

        # Multi-scale information fusion
        y1 = self.sigmoid(y1)
        y2 = self.sigmoid(y2)
        y3 = self.sigmoid(y3)
        y=y1+y2+y3
        y=self.sigmoid(y)
        return x * y.expand_as(x)

class PECA_BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(PECA_BasicBlock, self).__init__()
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

        ###
        self.se = P1ECALayer(planes)

        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = self.se(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class PECA_Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(PECA_Bottleneck, self).__init__()
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

        ###
        self.se = P1ECALayer(planes * self.expansion)

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

        out = self.se(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class PECA_ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None, nparts=2, attention=False, device='gpu'):
        super(PECA_ResNet, self).__init__()

        self.attention = attention
        self.device = device
        self.nparts = nparts

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

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
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        ###fc

        if self.attention:
            nfeatures = 512 * block.expansion
            nlocal_channels_norm = nfeatures // self.nparts
            reminder = nfeatures % self.nparts
            nlocal_channels_last = nlocal_channels_norm
            if reminder != 0:
                nlocal_channels_last = nlocal_channels_norm + reminder
            fc_list = []
            separations = []
            sep_node = 0
            for i in range(self.nparts):
                if i != self.nparts - 1:
                    sep_node += nlocal_channels_norm
                    fc_list.append(nn.Linear(nlocal_channels_norm, num_classes))
                    # separations.append(sep_node)
                else:
                    sep_node += nlocal_channels_last
                    fc_list.append(nn.Linear(nlocal_channels_last, num_classes))
                separations.append(sep_node)
            self.fclocal = nn.Sequential(*fc_list)
            self.separations = separations
            self.fc = nn.Linear(512 * block.expansion, num_classes)

        else:
            self.fc = nn.Linear(512 * block.expansion, num_classes)

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
                if isinstance(m, PECA_Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, PECA_Bottleneck):
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
        x = self.layer3(x)
        x = self.layer4(x)

        if self.attention:

            nsamples, nchannels, height, width = x.shape

            xview = x.view(nsamples, nchannels, -1)
            xnorm = xview.div(xview.norm(dim=-1, keepdim=True) + eps)
            xcosin = torch.bmm(xnorm, xnorm.transpose(-1, -2))

            attention_scores = []
            for i in range(self.nparts):
                if i == 0:
                    xx = x[:, :self.separations[i]]
                else:
                    xx = x[:, self.separations[i - 1]:self.separations[i]]
                xx_pool = self.avgpool(xx).flatten(1)
                attention_scores.append(self.fclocal[i](xx_pool))
            xlocal = torch.stack(attention_scores, dim=0)

            xmaps = x.clone().detach()

            # for global
            xpool = self.avgpool(x)
            xpool = torch.flatten(xpool, 1)
            xglobal = self.fc(xpool)

            return xglobal, xlocal, xcosin, xmaps
        else:
            # for original resnet outputs
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.fc(x)

            return x


def _ecanet(arch, block, layers, pretrained, progress, model_dir=None, **kwargs):
    model = PECA_ResNet(block, layers, **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls[arch], model_dir=model_dir))
        # state_dict = load_state_dict_from_url(model_urls[arch],
        #                                       progress=progress)
        # model.load_state_dict(state_dict)
    return model


def ecanet18(pretrained=False, progress=True, model_dir=None, **kwargs):
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _ecanet('ecanet18', PECA_BasicBlock, [2, 2, 2, 2], pretrained, progress, model_dir,
                   **kwargs)


def ecanet34(pretrained=False, progress=True, **kwargs):
    r"""ResNet-34 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _ecanet('ecanet34', PECA_BasicBlock, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)


def ecanet50(pretrained=False, progress=True, model_dir=None, **kwargs):
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _ecanet('ecanet50', PECA_Bottleneck, [3, 4, 6, 3], pretrained, progress, model_dir,
                   **kwargs)


def ecanet101(pretrained=False, progress=True, **kwargs):
    r"""ResNet-101 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _ecanet('ecanet101', PECA_Bottleneck, [3, 4, 23, 3], pretrained, progress,
                   **kwargs)


def ecanet152(pretrained=False, progress=True, **kwargs):
    r"""ResNet-152 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _ecanet('ecanet152', PECA_Bottleneck, [3, 8, 36, 3], pretrained, progress,
                   **kwargs)
