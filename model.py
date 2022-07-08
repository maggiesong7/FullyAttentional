import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from torch.nn import SyncBatchNorm


class BasicBlock(nn.Module):

    expansion = 1
    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None,
                 previous_dilation=1, norm_layer=SyncBatchNorm):

        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, 3, stride,
                               dilation, dilation, bias=False)
        self.bn1 = norm_layer(planes)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(planes, planes, 3, 1, previous_dilation,
                               dilation=previous_dilation, bias=False)
        self.bn2 = norm_layer(planes)
        self.relu = nn.ReLU()
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):

        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:

            identity = self.downsample(x)

        out = out + identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):

    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None,
                 previous_dilation=1, norm_layer=SyncBatchNorm):
        super(Bottleneck, self).__init__()

        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = norm_layer(planes)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(planes, planes, 3, stride,
                               dilation, dilation, bias=False)
        self.bn2 = norm_layer(planes)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU()
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):

        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out + identity
        out = self.relu(out)

        return out


class FLANetOutput(nn.Module):
    def __init__(self, in_plane, mid_plane, num_classes, norm_layer=SyncBatchNorm):
        super(FLANetOutput, self).__init__()
        self.conv = nn.Sequential(nn.Upsample(scale_factor=2),
                                  nn.Conv2d(in_plane, mid_plane, 3, stride=1, padding=1, bias=False),
                                  norm_layer(mid_plane),
                                  nn.ReLU(),
                                  nn.Dropout2d(0.1, False),
                                  nn.Conv2d(mid_plane, num_classes, kernel_size=1, stride=1, padding=0, bias=True))

    def forward(self, x):
        x = self.conv(x)
        return x


class FullyAttentionalBlock(nn.Module):
    def __init__(self, plane, norm_layer=SyncBatchNorm):
        super(FullyAttentionalBlock, self).__init__()
        self.conv1 = nn.Linear(plane, plane)
        self.conv2 = nn.Linear(plane, plane)
        self.conv = nn.Sequential(nn.Conv2d(plane, plane, 3, stride=1, padding=1, bias=False),
                                  norm_layer(plane),
                                  nn.ReLU())

        self.softmax = nn.Softmax(dim=-1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch_size, _, height, width = x.size()

        feat_h = x.permute(0, 3, 1, 2).contiguous().view(batch_size * width, -1, height)
        feat_w = x.permute(0, 2, 1, 3).contiguous().view(batch_size * height, -1, width)
        encode_h = self.conv1(F.avg_pool2d(x, [1, width]).view(batch_size, -1, height).permute(0, 2, 1).contiguous())
        encode_w = self.conv2(F.avg_pool2d(x, [height, 1]).view(batch_size, -1, width).permute(0, 2, 1).contiguous())

        energy_h = torch.matmul(feat_h, encode_h.repeat(width, 1, 1))
        energy_w = torch.matmul(feat_w, encode_w.repeat(height, 1, 1))
        full_relation_h = self.softmax(energy_h)  # [b*w, c, c]
        full_relation_w = self.softmax(energy_w)

        full_aug_h = torch.bmm(full_relation_h, feat_h).view(batch_size, width, -1, height).permute(0, 2, 3, 1)
        full_aug_w = torch.bmm(full_relation_w, feat_w).view(batch_size, height, -1, width).permute(0, 2, 1, 3)
        out = self.gamma * (full_aug_h + full_aug_w) + x
        out = self.conv(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, n_classes=19, dilated=True, deep_stem=True,
                 zero_init_residual=False, norm_layer=SyncBatchNorm):

        self.inplanes = 128 if deep_stem else 64
        super(ResNet, self).__init__()

        if deep_stem:
            self.conv1 = nn.Sequential(
                nn.Conv2d(3, 64, 3, 2, 1, bias=False),
                norm_layer(64),
                nn.ReLU(),
                nn.Conv2d(64, 64, 3, 1, 1, bias=False),
                norm_layer(64),
                nn.ReLU(),
                nn.Conv2d(64, 128, 3, 1, 1, bias=False)
            )
        else:
            self.conv1 = nn.Conv2d(3, 64, 7, 2, 3, bias=False)

        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(3, 2, 1)
        self.layer1 = self._make_layer(block, 64, layers[0], norm_layer=norm_layer)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, norm_layer=norm_layer)

        if dilated:
            self.layer3 = self._make_layer(block, 256, layers[2], stride=1, dilation=2, norm_layer=norm_layer)
            self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilation=4, norm_layer=norm_layer)
        else:
            self.layer3 = self._make_layer(block, 256, layers[2], stride=2, norm_layer=norm_layer)
            self.layer4 = self._make_layer(block, 512, layers[3], stride=2, norm_layer=norm_layer)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def load_pretrained(self, path):
        pre_trained = torch.load(path, map_location='cpu')
        state_dict = self.state_dict()
        for key, weights in pre_trained.items():
            if key in state_dict:
                state_dict[key].copy_(weights)

        self.load_state_dict(state_dict)

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1, norm_layer=SyncBatchNorm):
        downsample = None

        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, 1, stride, bias=False),
                norm_layer(planes * block.expansion),
            )

        layers = []

        layers.append(block(self.inplanes, planes, stride, dilation=dilation, downsample=downsample,
                            previous_dilation=dilation, norm_layer=norm_layer))

        self.inplanes = planes * block.expansion

        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation,
                                previous_dilation=dilation, norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        h, w = x.size()[2:]

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x1 = self.layer2(x)
        x2 = self.layer3(x1)
        x3 = self.layer4(x2)

        return x1, x2, x3


class FLANet(nn.Module):
    def __init__(self, block, layers, n_classes=19, dilated=True, deep_stem=True,
                 zero_init_residual=False, norm_layer=SyncBatchNorm):

        super(FLANet, self).__init__()

        self.backbone = ResNet(block, layers, n_classes=n_classes, dilated=dilated, deep_stem=deep_stem,
                               zero_init_residual=zero_init_residual, norm_layer=norm_layer)

        self.conv = nn.Sequential(nn.Conv2d(2048, 512, 3, stride=1, padding=1, bias=False),
                                  norm_layer(512),
                                  nn.ReLU(),
                                  nn.Conv2d(512, 512, 3, stride=1, padding=1, bias=False),
                                  norm_layer(512),
                                  nn.ReLU())

        self.fully = FullyAttentionalBlock(512)

        self.conv_out1 = FLANetOutput(2048 + 512, 512, n_classes)
        self.conv_out2 = FLANetOutput(512, 64, n_classes)
        self.conv_out3 = FLANetOutput(1024, 64, n_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=1)
            elif isinstance(m, SyncBatchNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        h, w = x.size()[2:]

        x1, x2, x3 = self.backbone(x)

        full_feats = self.fully(self.conv(x3))

        full_feats = self.conv_out1(torch.cat([x3, full_feats], 1))
        aux1 = self.conv_out2(x1)
        aux2 = self.conv_out3(x2)

        output = F.interpolate(full_feats, (h, w), mode='bilinear', align_corners=True)
        aux1 = F.interpolate(aux1, (h, w), mode='bilinear', align_corners=True)
        aux2 = F.interpolate(aux2, (h, w), mode='bilinear', align_corners=True)

        outs = [output, aux1, aux2]

        return outs


def compile_model(n_classes=19, pre_train=True, path='./pretrained_ckpt/resnet101-deep.pth'):
    model = FLANet(Bottleneck, [3, 4, 23, 3], n_classes)

    if pre_train:
        model.backbone.load_pretrained(path)

    return model


