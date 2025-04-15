# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmengine.model import BaseModule
from torch import Tensor

from mmdet.registry import MODELS
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig
from ..layers.se_layer import ChannelAttention

"""Feature Optimization Network for Enhancement (FONE)."""

@MODELS.register_module()
class FONE(BaseModule):
    """Feature-Only Neck with Enhanced attention.

    This neck takes a single feature map from the backbone and generates
    multiple feature maps with different scales. It also applies a lightweight
    attention mechanism to enhance the features.

    Args:
        in_channels (int): Number of input channels from backbone.
        out_channels (int): Number of output channels for each scale.
        num_outs (int): Number of output scales.
        start_level (int): The level of input feature map to use.
        conv_cfg (dict, optional): Config dict for convolution layer.
            Default: None.
        norm_cfg (dict, optional): Config dict for normalization layer.
            Default: dict(type='BN').
        act_cfg (dict, optional): Config dict for activation layer in ConvModule.
            Default: dict(type='ReLU').
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: dict(type='Kaiming', layer='Conv2d').
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 num_outs: int = 3,
                 start_level: int = 0,
                 conv_cfg: OptConfigType = None,
                 norm_cfg: ConfigType = dict(type='BN'),
                 act_cfg: ConfigType = dict(type='ReLU'),
                 init_cfg: OptMultiConfig = dict(
                     type='Kaiming', layer='Conv2d')) -> None:
        super().__init__(init_cfg=init_cfg)
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_outs = num_outs
        self.start_level = start_level
        
        # Feature extraction modules
        self.multi_scale_convs = nn.ModuleList()
        
        # First conv to adjust channel dimension
        self.input_conv = ConvModule(
            in_channels,
            out_channels,
            kernel_size=1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        
        # Generate multi-scale features
        for i in range(num_outs):
            # Different kernel sizes for different scales
            kernel_size = 3 + i * 2  # 3, 5, 7, ...
            padding = kernel_size // 2
            
            conv = ConvModule(
                out_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=padding,
                stride=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg)
            
            self.multi_scale_convs.append(conv)
        
        # Lightweight attention modules
        self.attention_modules = nn.ModuleList()
        for _ in range(num_outs):
            self.attention_modules.append(ChannelAttention(out_channels))

    def forward(self, inputs: Tuple[Tensor]) -> Tuple[Tensor]:
        """Forward function.

        Args:
            inputs (tuple[Tensor]): Features from the backbone.

        Returns:
            tuple[Tensor]: Multi-scale features with attention applied.
        """
        # Take the feature map at the specified level
        x = inputs[self.start_level]
        
        # Adjust channel dimension
        x = self.input_conv(x)
        
        # Generate multi-scale features with attention
        outs = []
        for i in range(self.num_outs):
            # Apply different scale convolution
            feat = self.multi_scale_convs[i](x)
            
            # Apply attention mechanism
            feat = self.attention_modules[i](feat)
            
            outs.append(feat)
        
        return tuple(outs)