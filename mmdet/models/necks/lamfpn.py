import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule, DepthwiseSeparableConvModule
from mmengine.model import BaseModule
from torch.utils.checkpoint import checkpoint
from mmdet.registry import MODELS
from mmdet.models.necks.fpn import FPN


@MODELS.register_module()
class LAMFPN(BaseModule):
    """
    LAMFPN (Local Attention Module Feature Pyramid Network)
    
    This neck is designed to be a drop-in replacement for BiFPN with enhanced 
    attention mechanisms and feature fusion.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 num_stages=3,  # 与BiFPN兼容的参数
                 start_level=0,
                 end_level=-1,
                 num_outs=5,  # 默认输出5个特征层，与BiFPN一致
                 add_extra_convs=False,
                 relu_before_extra_convs=False,
                 no_norm_on_lateral=False,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', momentum=0.01, eps=1e-3),  # 与BiFPN默认值一致
                 act_cfg=dict(type='ReLU'),
                 upsample_cfg=dict(mode='nearest'),
                 init_cfg=dict(type='Xavier', layer='Conv2d', distribution='uniform'),
                 # LAMFPN特有参数
                 apply_lam_levels=[1, 2],
                 use_dw_conv=True,
                 channel_reduction=4,
                 skip_attention_thresh=0.05,
                 use_modern_norm=True,
                 use_modern_act='silu',
                 attention_type='dual',
                 use_cross_layer_attention=False,
                 use_checkpoint=False,
                 # BiFPN特有参数，为了兼容性保留
                 epsilon=1e-4,
                 apply_bn_for_resampling=True,
                 conv_bn_act_pattern=False,
                 **kwargs):
        super(LAMFPN, self).__init__(init_cfg)
        assert len(in_channels) >= 3, "LAMFPN requires at least 3 input channels"
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_stages = num_stages
        self.start_level = start_level
        self.end_level = end_level
        self.num_outs = num_outs
        self.upsample_cfg = upsample_cfg
        self.use_checkpoint = use_checkpoint
        
        # 处理end_level
        if self.end_level == -1:
            self.backbone_end_level = len(in_channels)
            assert num_outs >= len(in_channels) - start_level
        else:
            self.backbone_end_level = end_level
            assert end_level <= len(in_channels)
            assert num_outs == end_level - start_level
            
        # 设置现代化的归一化和激活函数
        self.modern_norm_cfg = dict(type='GN', num_groups=8, requires_grad=True) if use_modern_norm else norm_cfg
        self.modern_act_cfg = dict(type='SiLU') if use_modern_act == 'silu' else dict(
            type='Mish') if use_modern_act == 'mish' else act_cfg
        
        # 初始化lateral卷积
        self.lateral_convs = nn.ModuleList()
        for i in range(self.start_level, self.backbone_end_level):
            l_conv = ConvModule(
                in_channels[i],
                out_channels,
                1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg if not no_norm_on_lateral else None,
                act_cfg=act_cfg,
                inplace=False)
            self.lateral_convs.append(l_conv)
            
        # 初始化FPN卷积
        self.fpn_convs = nn.ModuleList()
        for i in range(self.num_outs):
            fpn_conv = ConvModule(
                out_channels,
                out_channels,
                3,
                padding=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                inplace=False)
            self.fpn_convs.append(fpn_conv)
            
        # 额外层级的处理
        self.add_extra_convs = add_extra_convs
        self.relu_before_extra_convs = relu_before_extra_convs
        
        # 初始化多阶段结构
        self.stages = nn.ModuleList()
        for _ in range(self.num_stages):
            stage = LAMFPNStage(
                out_channels,
                out_channels,
                apply_lam_levels=apply_lam_levels,
                use_dw_conv=use_dw_conv,
                channel_reduction=channel_reduction,
                skip_attention_thresh=skip_attention_thresh,
                attention_type=attention_type,
                use_cross_layer_attention=use_cross_layer_attention,
                conv_cfg=conv_cfg,
                norm_cfg=self.modern_norm_cfg,
                act_cfg=self.modern_act_cfg,
                use_checkpoint=use_checkpoint
            )
            self.stages.append(stage)
            
        # 存储其他参数
        self.apply_lam_levels = apply_lam_levels
        self.skip_attention_thresh = skip_attention_thresh
        self.attention_type = attention_type
        self.use_cross_layer_attention = use_cross_layer_attention

    def forward(self, inputs):
        # 确保输入特征数量正确
        assert len(inputs) == len(self.in_channels)
        
        # 选择需要的输入特征
        feats = inputs[self.start_level:self.backbone_end_level]
        
        # 调整通道数
        laterals = [
            lateral_conv(feat)
            for feat, lateral_conv in zip(feats, self.lateral_convs)
        ]
        
        # 如果需要更多输出层级，添加额外的特征图
        if self.num_outs > len(laterals):
            # 添加P6
            if self.add_extra_convs and self.add_extra_convs == 'on_input':
                p6 = F.max_pool2d(inputs[self.backbone_end_level - 1], 1, stride=2)
            else:
                p6 = F.max_pool2d(laterals[-1], 1, stride=2)
            laterals.append(p6)
            
            # 添加P7
            p7 = F.max_pool2d(laterals[-1], 1, stride=2)
            laterals.append(p7)
        
        # 通过多个LAMFPN阶段处理特征
        features = laterals
        for stage in self.stages:
            features = stage(features)
        
        # 应用FPN卷积
        outs = [
            self.fpn_convs[i](feat)
            for i, feat in enumerate(features)
        ]
        
        return tuple(outs)


class LAMFPNStage(BaseModule):
    """
    单个LAMFPN阶段，替代BiFPN的单个阶段
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 apply_lam_levels=[1, 2],
                 use_dw_conv=True,
                 channel_reduction=4,
                 skip_attention_thresh=0.05,
                 attention_type='dual',
                 use_cross_layer_attention=False,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU'),
                 use_checkpoint=True,
                 init_cfg=dict(type='Xavier', layer='Conv2d', distribution='uniform')):
        super(LAMFPNStage, self).__init__(init_cfg)
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.apply_lam_levels = apply_lam_levels
        self.use_checkpoint = use_checkpoint
        self.skip_attention_thresh = skip_attention_thresh
        
        num_levels = 5  # 与BiFPN保持一致，处理5个特征层 (P3-P7)
        
        # LAM模块初始化
        self.lam_modules = nn.ModuleList([
            LAMModule(
                out_channels,
                out_channels,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                use_dw_conv=use_dw_conv,
                channel_reduction=channel_reduction
            ) if i in self.apply_lam_levels else None
            for i in range(num_levels - 1)
        ])
        
        # 上采样和下采样路径
        self.top_down_blocks = nn.ModuleList([
            ConvModule(
                out_channels,
                out_channels,
                3,
                padding=1,
                groups=out_channels if use_dw_conv else 1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg
            ) for _ in range(num_levels - 1)
        ])
        
        self.bottom_up_blocks = nn.ModuleList([
            ConvModule(
                out_channels,
                out_channels,
                3,
                padding=1,
                groups=out_channels if use_dw_conv else 1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg
            ) for _ in range(num_levels - 1)
        ])
        
        # 注意力模块
        self.attention_type = attention_type
        self.feature_attention = nn.ModuleList([
            DualAttention(out_channels, reduction=16, use_dw_conv=use_dw_conv)
            if i in [1, 2] and attention_type == 'dual'
            else None
            for i in range(num_levels)
        ])
        
        # 跨层注意力
        self.use_cross_layer_attention = use_cross_layer_attention
        self.cross_attention = CrossLayerAttention(
            out_channels, num_levels,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg
        ) if use_cross_layer_attention else None

    def forward(self, inputs):
        if self.use_checkpoint and self.training:
            return checkpoint(self._forward_impl, inputs, use_reentrant=False)
        return self._forward_impl(inputs)
    
    def _forward_impl(self, inputs):
        # 确保输入特征数量正确
        assert len(inputs) >= 3, "LAMFPN stage requires at least 3 input features"
        
        features = list(inputs)
        num_levels = len(features)
        
        # 自顶向下路径 (类似BiFPN的top-down path)
        top_down_features = [features[-1]]  # 从最顶层开始
        for i in range(num_levels - 2, -1, -1):
            feat = features[i]
            feat_size = feat.shape[2:]
            
            # 上采样上层特征
            higher_feat = F.interpolate(
                top_down_features[0],
                size=feat_size,
                mode='bilinear', 
                align_corners=False
            )
            
            # 应用LAM模块或简单相加
            if i in self.apply_lam_levels and self.lam_modules[i] is not None:
                if self.use_checkpoint and self.training:
                    # TODO: checkpoint 的使用方式
                    fused = checkpoint(
                        self.lam_modules[i],
                        feat,
                        higher_feat,
                        use_reentrant=False
                    )
                else:
                    fused = self.lam_modules[i](feat, higher_feat)
            else:
                fused = feat + higher_feat
            
            # 应用卷积
            processed = self.top_down_blocks[i](fused)
            top_down_features.insert(0, processed)
        
        # 自底向上路径 (类似BiFPN的bottom-up path)
        bottom_up_features = [top_down_features[0]]
        for i in range(1, num_levels):
            feat = top_down_features[i]
            feat_size = feat.shape[2:]
            
            # 下采样下层特征
            if i < num_levels - 1:
                lower_feat = F.adaptive_max_pool2d(
                    bottom_up_features[-1],
                    output_size=(feat_size[0], feat_size[1])
                )
                
                # 应用LAM模块或简单相加
                if (i-1) in self.apply_lam_levels and self.lam_modules[i-1] is not None:
                    if self.use_checkpoint and self.training:
                        fused = checkpoint(
                            self.lam_modules[i-1],
                            feat,
                            lower_feat,
                            use_reentrant=False
                        )
                    else:
                        fused = self.lam_modules[i-1](feat, lower_feat)
                else:
                    fused = feat + lower_feat
                
                # 应用卷积
                processed = self.bottom_up_blocks[i-1](fused)
            else:
                # 最顶层没有下层特征
                processed = self.bottom_up_blocks[i-1](feat)
            
            bottom_up_features.append(processed)
        
        # 应用注意力机制
        enhanced_features = []
        for i, feat in enumerate(bottom_up_features):
            if self.feature_attention[i] is not None:
                with torch.no_grad():
                    # importance = torch.mean(F.adaptive_avg_pool2d(feat, 1))
                    importance = torch.mean(F.adaptive_avg_pool2d(feat, 1).flatten())
                if importance > self.skip_attention_thresh:
                    if self.use_checkpoint and self.training:
                        enhanced_features.append(
                            checkpoint(self.feature_attention[i], feat, use_reentrant=False))
                    else:
                        enhanced_features.append(
                            self.feature_attention[i](feat))
                else:
                    enhanced_features.append(feat)
            else:
                enhanced_features.append(feat)
        
        # 跨层注意力融合
        if self.cross_attention is not None:
            if self.use_checkpoint and self.training:
                enhanced_features = checkpoint(
                    self.cross_attention, enhanced_features, use_reentrant=False)
            else:
                enhanced_features = self.cross_attention(enhanced_features)
                
        return enhanced_features


class LAMModule(BaseModule):
    def __init__(self,
                 in_channels,
                 out_channels,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU'),
                 use_dw_conv=True,
                 channel_reduction=4,
                 init_cfg=dict(
                     type='Xavier', layer='Conv2d', distribution='uniform')):
        super(LAMModule, self).__init__(init_cfg)

        reduced_channels = in_channels // channel_reduction
        conv_module = DepthwiseSeparableConvModule if use_dw_conv else ConvModule

        self.attention_conv = conv_module(
            in_channels * 2,
            reduced_channels,
            kernel_size=3,
            padding=1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)

        self.attention_weights = nn.Conv2d(
            reduced_channels, 2, kernel_size=1, padding=0
        )
        self.softmax = nn.Softmax(dim=1)
        self.fusion_conv = conv_module(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg
        )

    def forward(self, target_feat, source_feat):
        concat_feat = torch.cat([target_feat, source_feat], dim=1)
        attention_feat = self.attention_conv(concat_feat)
        attention_weights = self.attention_weights(attention_feat)
        attention_weights = self.softmax(attention_weights)

        fused_feat = (
                attention_weights[:, 0:1] * target_feat +
                attention_weights[:, 1:2] * source_feat
        )
        return self.fusion_conv(fused_feat)


class DualAttention(BaseModule):
    def __init__(self,
                 channels,
                 reduction=16,
                 use_dw_conv=True,
                 init_cfg=dict(
                     type='Xavier', layer='Conv2d', distribution='uniform')):
        super(DualAttention, self).__init__(init_cfg)

        self.shared_mlp = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False)
        )

        if use_dw_conv:
            self.spatial_attention = nn.Sequential(
                nn.Conv2d(2, 1, 1, bias=False),
                nn.Conv2d(1, 1, 7, padding=3, groups=1, bias=False),
                nn.Sigmoid()
            )
        else:
            self.spatial_attention = nn.Sequential(
                nn.Conv2d(2, 1, 7, padding=3, bias=False),
                nn.Sigmoid()
            )

    def forward(self, x):
        # 通道注意力 - 使用共享MLP
        avg_pool = F.adaptive_avg_pool2d(x, 1)
        max_pool = F.adaptive_max_pool2d(x, 1)

        avg_out = self.shared_mlp(avg_pool)
        max_out = self.shared_mlp(max_pool)

        channel_att = torch.sigmoid(avg_out + max_out)
        x = x * channel_att

        # 空间注意力
        spatial_avg = torch.mean(x, dim=1, keepdim=True)
        spatial_max, _ = torch.max(x, dim=1, keepdim=True)
        spatial_feat = torch.cat([spatial_avg, spatial_max], dim=1)
        spatial_att = self.spatial_attention(spatial_feat)

        return x * spatial_att


class CrossLayerAttention(BaseModule):
    def __init__(self,
                 channels,
                 num_levels,
                 norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
                 act_cfg=dict(type='SiLU'),
                 init_cfg=dict(
                     type='Xavier', layer='Conv2d', distribution='uniform')):
        super().__init__(init_cfg)

        self.num_levels = num_levels
        self.channels = channels

        # 共享特征变换
        self.shared_transform = ConvModule(
            channels, channels, 1,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg
        )

        # 共享权重生成器
        self.shared_weight_gen = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, num_levels - 1, 1),
            nn.Sigmoid()
        )

        # 特征融合模块
        self.fuse_conv = ConvModule(
            channels, channels, 3,
            padding=1,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg
        )

    def forward(self, features):
        batch_size = features[0].shape[0]
        device = features[0].device

        # 特征变换
        with torch.set_grad_enabled(self.training):
            transformed = [self.shared_transform(feat) for feat in features]

        outputs = []
        for level_idx, feat in enumerate(features):
            other_feats = []

            # 收集其他层级特征
            for idx in range(self.num_levels):
                if idx != level_idx and idx < len(features):
                    # 使用内存效率更高的方式进行特征插值
                    with torch.no_grad():
                        resized = F.interpolate(
                            transformed[idx],
                            size=feat.shape[2:],
                            mode='bilinear',
                            align_corners=False
                        )
                    other_feats.append(resized)

            if other_feats:
                # 生成权重
                weights = self.shared_weight_gen(feat)
                assert len(other_feats) <= weights.shape[1], (
                    f"Number of other features ({len(other_feats)}) "
                    f"should not exceed the number of weights "
                    f"({weights.shape[1]})."
                )
                # 确保权重维度与特征数量匹配
                weights = weights[:, :len(other_feats)]

                # 流式处理特征融合
                fused = torch.zeros_like(feat)
                for i, other_feat in enumerate(other_feats):
                    fused.add_(other_feat * weights[:, i:i + 1])

                # 残差连接
                output = self.fuse_conv(feat + fused)
            else:
                output = feat

            outputs.append(output)

            # 清理中间变量
            del other_feats
            if 'fused' in locals():
                del fused

        return outputs
