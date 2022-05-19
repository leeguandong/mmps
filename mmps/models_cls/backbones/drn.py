'''
@Time    : 2022/5/19 15:31
@Author  : leeguandon@gmail.com
'''
import math
import torch.nn as nn
from mmcv.cnn import (ConvModule, build_activation_layer, build_conv_layer,
                      build_norm_layer, constant_init)
from mmcv.utils.parrots_wrapper import _BatchNorm

from mmcls.models.builder import BACKBONES
from mmcls.models.backbones.base_backbone import BaseBackbone
from mmcls.models.backbones.resnet import BasicBlock, Bottleneck, get_expansion

eps = 1.0e-5


@BACKBONES.register_module()
class DRN(BaseBackbone):
    arch_settings = {
        'c': {26: (BasicBlock, (1, 1, 2, 2, 2, 2, 1, 1)),
              42: (BasicBlock, (1, 1, 3, 4, 6, 3, 1, 1)),
              58: (BasicBlock, (1, 1, 3, 4, 6, 3, 1, 1))},
        'd': {22: (BasicBlock, (1, 1, 2, 2, 2, 2, 1, 1)),
              24: (BasicBlock, (1, 1, 2, 2, 2, 2, 2, 2)),
              38: (BasicBlock, (1, 1, 3, 4, 6, 3, 1, 1)),
              40: (BasicBlock, (1, 1, 3, 4, 6, 3, 2, 2)),
              54: (Bottleneck, (1, 1, 3, 4, 6, 3, 1, 1)),
              56: (Bottleneck, (1, 1, 3, 4, 6, 3, 2, 2)),
              105: (Bottleneck, (1, 1, 3, 4, 23, 3, 1, 1)),
              107: (Bottleneck, (1, 1, 3, 4, 23, 3, 2, 2))}
    }

    def __init__(self,
                 arch,
                 depth,
                 in_channels=3,
                 stem_channels=16,
                 channels=(16, 32, 64, 128, 256, 512, 512, 512),
                 out_indices=(3,),
                 style='pytorch',
                 deep_stem=False,
                 avg_down=False,
                 frozen_stages=-1,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 norm_eval=False,
                 with_cp=False,
                 zero_init_residual=True,
                 init_cfg=[
                     dict(type='Kaiming', layer=['Conv2d']),
                     dict(
                         type='Constant',
                         val=1,
                         layer=['_BatchNorm', 'GroupNorm'])
                 ],
                 drop_path_rate=0.0
                 ):
        super(DRN, self).__init__(init_cfg)
        block, layers = self.arch_settings[arch][depth]
        self.deep_stem = deep_stem
        self.avg_down = avg_down
        self.frozen_stages = frozen_stages
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.with_cp = with_cp
        self.norm_eval = norm_eval
        self.zero_init_residual = zero_init_residual
        self.out_indices = out_indices
        self.inplanes = channels[0]
        self.out_dim = channels[-1]

        self._make_stem_layer(in_channels, stem_channels)

        if arch == 'c':
            self.layer1 = self._make_layer(
                BasicBlock, channels[0], layers[0], stride=1)
            self.layer2 = self._make_layer(
                BasicBlock, channels[1], layers[1], stride=2)
        elif arch == 'd':
            self.layer1 = self._make_conv_layers(
                channels[0], layers[0], stride=1)
            self.layer2 = self._make_conv_layers(
                channels[1], layers[1], stride=2)

        self.layer3 = self._make_layer(block, channels[2], layers[2], stride=2)
        self.layer4 = self._make_layer(block, channels[3], layers[3], stride=2)
        self.layer5 = self._make_layer(block, channels[4], layers[4], dilation=2, new_level=False)
        self.layer6 = None if layers[5] == 0 else \
            self._make_layer(block, channels[5], layers[5], dilation=4,
                             new_level=False)

        if arch == 'c':
            self.layer7 = None if layers[6] == 0 else \
                self._make_layer(BasicBlock, channels[6], layers[6], dilation=2,
                                 new_level=False, residual=False)
            self.layer8 = None if layers[7] == 0 else \
                self._make_layer(BasicBlock, channels[7], layers[7], dilation=1,
                                 new_level=False, residual=False)
        elif arch == 'd':
            self.layer7 = None if layers[6] == 0 else \
                self._make_conv_layers(channels[6], layers[6], dilation=2)
            self.layer8 = None if layers[7] == 0 else \
                self._make_conv_layers(channels[7], layers[7], dilation=1)

        self._freeze_stages()
        self.feat_dim = self.out_dim

    @property
    def norm1(self):
        return getattr(self, self.norm1_name)

    def _make_stem_layer(self, in_channels, stem_channels):
        if self.deep_stem:
            self.stem = nn.Sequential(
                ConvModule(
                    in_channels,
                    stem_channels // 2,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    inplace=True),
                ConvModule(
                    stem_channels // 2,
                    stem_channels // 2,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    inplace=True),
                ConvModule(
                    stem_channels // 2,
                    stem_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    inplace=True))
        else:
            self.conv1 = build_conv_layer(
                self.conv_cfg,
                in_channels,
                stem_channels,
                kernel_size=7,
                stride=1,
                padding=3,
                bias=False)
            self.norm1_name, norm1 = build_norm_layer(
                self.norm_cfg, stem_channels, postfix=1)
            self.add_module(self.norm1_name, norm1)
            self.relu = nn.ReLU(inplace=True)

    def _make_layer(self, block, planes, num_blocks, stride=1, dilation=1,
                    new_level=True):
        expansion = get_expansion(block)
        assert dilation == 1 or dilation % 2 == 0
        downsample = None
        if stride != 1 or self.inplanes != planes * expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                stride,
                downsample,
                dilation=(1, 1) if dilation == 1 else (
                    dilation // 2 if new_level else dilation, dilation)))
        self.inplanes = planes * expansion
        for i in range(1, num_blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    dilation=(dilation, dilation)))
        return nn.Sequential(*layers)

    def _make_conv_layers(self, channels, convs, stride=1, dilation=1):
        modules = []
        for i in range(convs):
            modules.extend([
                nn.Conv2d(self.inplanes, channels, kernel_size=3,
                          stride=stride if i == 0 else 1,
                          padding=dilation, bias=False, dilation=dilation),
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True)])
            self.inplanes = channels
        return nn.Sequential(*modules)

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            if self.deep_stem:
                self.stem.eval()
                for param in self.stem.parameters():
                    param.requires_grad = False
            else:
                self.norm1.eval()
                for m in [self.conv1, self.norm1]:
                    for param in m.parameters():
                        param.requires_grad = False

        for i in range(1, self.frozen_stages + 1):
            m = getattr(self, f'layer{i}')
            m.eval()
            for param in m.parameters():
                param.requires_grad = False

    def init_weights(self):
        super(DRN, self).init_weights()

        if (isinstance(self.init_cfg, dict)
            and self.init_cfg['type'] == 'Pretrained'):
            # Suppress zero_init_residual if use pretrained model.
            return

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        if self.zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    constant_init(m.norm3, 0)
                elif isinstance(m, BasicBlock):
                    constant_init(m.norm2, 0)

    def forward(self, x):
        y = list()

        if self.deep_stem:
            x = self.stem(x)
        else:
            x = self.conv1(x)
            x = self.norm1(x)
            x = self.relu(x)

        x = self.layer1(x)
        y.append(x)
        x = self.layer2(x)
        y.append(x)

        x = self.layer3(x)
        y.append(x)

        x = self.layer4(x)
        y.append(x)

        x = self.layer5(x)
        y.append(x)

        if self.layer6 is not None:
            x = self.layer6(x)
            y.append(x)

        if self.layer7 is not None:
            x = self.layer7(x)
            y.append(x)

        if self.layer8 is not None:
            x = self.layer8(x)
            y.append(x)

        return tuple(y[self.out_indices:])

    def train(self, mode=True):
        super(DRN, self).train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                # trick: eval have effect on BatchNorm only
                if isinstance(m, _BatchNorm):
                    m.eval()


if __name__ == "__main__":
    import torch

    model = DRN(arch='d', depth=105)
    print(model)
    with torch.no_grad():
        pass
