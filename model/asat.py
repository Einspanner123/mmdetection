import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.registry import MODELS

class PatchEmbed(nn.Module):
    def __init__(self, patch_size=16, in_channels=3, embed_dim=768):
        super(PatchEmbed, self).__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(
            in_channels, 
            embed_dim, 
            kernel_size=patch_size, 
            stride=patch_size
        )

    def forward(self, x):
        # 计算patch数量
        _, _, H, W = x.shape
        assert H % self.patch_size == 0 and W % self.patch_size == 0
        num_patches = (H // self.patch_size) * (W // self.patch_size)

        # 进行卷积投射
        x = self.proj(x) # B, C, H, W => B, embed_dim, H/patch_size, W/patch_size

        # 重塑张量
        x = x.permute(0, 2, 3, 1) # B, H/patch_size, W/patch_size, embed_dim

        return x
    
class FeedForwardAmplifier(nn.Module):
    def __init__(
        self, 
        in_channels, 
        hidden_channels, 
        out_channels, 
        upsample_factor=2,
        dropout=None):
        super(FeedForwardAmplifier, self).__init__()
        self.linear_proj = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout) if dropout is not None else nn.Identity(),
            # nn.Linear(hidden_channels, out_channels),
            # nn.Dropout(p=dropout) if dropout is not None else nn.Identity(),
        )
        self.deconv = nn.ConvTranspose2d(
            in_channels=hidden_channels,
            out_channels=out_channels,
            kernel_size=upsample_factor,
            stride=upsample_factor,
            padding=0
        )
        self.norm = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=dropout) if dropout is not None else nn.Identity()

    def forward(self, x):
        x = self.linear_proj(x.permute(0, 2, 3, 1))
        x = self.deconv(x.permute(0, 3, 1, 2))
        x = self.norm(x)
        x = self.act(x)
        x = self.dropout(x)
        return x
                
class MLP(nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_channels,
        out_channels, 
        bias=True,
        dropout=None):
        super(MLP, self).__init__()
        self.ffn = nn.Sequential(
            nn.Linear(in_channels, hidden_channels, bias=bias),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout) if dropout is not None else nn.Identity(),
            nn.Linear(hidden_channels, out_channels, bias=bias),
            nn.Dropout(p=dropout) if dropout is not None else nn.Identity(),
        )

    def forward(self, x):
        return self.ffn(x)

class MSA(nn.Module):
    """
    注意力机制模块，用于增强模型对不同位置特征的关注
    这有助于模型学习局部特征，同时减少了参数量
    """
    def __init__(
        self, 
        in_channels, 
        num_heads=4, 
        qkv_bias=True, 
        use_rel_pos=False):
        super(MSA, self).__init__()
        self.num_heads = num_heads
        head_dim = in_channels // num_heads
        self.scale = (head_dim ** -0.5)
        self.qkv_proj = nn.Linear(in_channels, in_channels * 3, bias=qkv_bias)
        self.out_proj = nn.Linear(in_channels, in_channels)

    def forward(self, x):
        B, H, W, _ = x.shape
        qkv = self.qkv_proj(x).reshape(B, H*W, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4) # 3, B, heads, H*W, head_dim
        q, k, v = qkv.reshape(3, B*self.num_heads, H*W, -1).unbind(0) # B*heads, H*W, head_dim
        # 计算注意力
        attn = torch.softmax((q*self.scale) @ k.transpose(-2, -1), dim=-1) # B*heads, H*W, H*W
        
        # (B*heads, H*W, head_dim) => (B, heads, H, W, head_dim) => (B, H, W, heads, head_dim) => (B, H, W, C)
        x = (attn @ v).view(B, self.num_heads, H, W, -1).permute(0, 2, 3, 1, 4).reshape(B, H, W, -1)

        return self.out_proj(x)

class GroupSelfAttention(nn.Module):
    def __init__(self, in_channels, attn_heads=4,  repeats=2, dropout=None):
        super(GroupSelfAttention, self).__init__()
        
        self.norm_in = nn.LayerNorm(in_channels)
        self.norm_out = nn.LayerNorm(in_channels)
        assert in_channels % attn_heads == 0
        
        self.attns = nn.ModuleList([
            MSA(
                in_channels=in_channels,
                num_heads=attn_heads,
                qkv_bias=True,
                use_rel_pos=False
            ) for _ in range(repeats)
        ])

        self.MLP = MLP(
            in_channels=in_channels,
            hidden_channels=in_channels * 4,
            out_channels=in_channels,
            bias=True,
            dropout=dropout
        )
    
    def forward(self, x):
        B, C, H, W = x.shape
        
        x = self.norm_in(x.permute(0, 2, 3, 1))
        out = x
        for attn in self.attns:
            out = attn(out) + out
        # 合并所有组的输出
        # output = torch.cat(output_features, dim=-1) # B, H, W, C
        out = self.MLP(self.norm_out(out) + out)
        
        return out.permute(0, 3, 1, 2)

class GroupEncoder(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        groups=4,
        dropout=None
        ):
        super(GroupEncoder, self).__init__()
        self.groups = groups
        assert out_channels % groups == 0
        self.lateral_connect = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1), # B, C, H, W => B, out_channels, H, W
            nn.BatchNorm2d(out_channels),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1), # B, out_channels, H, W => B, out_channels, H, W
            nn.BatchNorm2d(out_channels)
        )
        self.dilated_blocks = nn.Sequential(*[
            BottleNeck(
                out_channels, 
                out_channels // 4,
                dilation=dilation,
                dropout=dropout
            ) for dilation in (2, 4, 6, 8)
        ])
        # 空间注意力机制
        # self.spatial_attention = nn.ModuleList([
        #     nn.Sequential(
        #         nn.AdaptiveAvgPool2d(1),
        #         nn.Conv2d(out_channels, out_channels // 4, kernel_size=1),
        #         nn.ReLU(inplace=True),
        #         nn.Conv2d(out_channels // 4, out_channels, kernel_size=1),
        #         nn.Sigmoid()
        #     ) for _ in range(groups)
        # ])
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        out = self.dilated_blocks(self.lateral_connect(x))
        return out

class BottleNeck(nn.Module):
    def __init__(self, in_channels, mid_channels, dilation, dropout=None):
        super(BottleNeck, self).__init__()
        self.in_layer = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout) if dropout is not None else nn.Identity()
        )
        self.mid_layer = nn.Sequential(
            nn.Conv2d(
                mid_channels, 
                mid_channels, 
                kernel_size=3,
                dilation=dilation, 
                padding=dilation,
                bias=False
            ),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout) if dropout is not None else nn.Identity()
        )
        self.out_layer = nn.Sequential(
            nn.Conv2d(mid_channels, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout) if dropout is not None else nn.Identity()
        )
    def forward(self, x):
        shortcut = x
        out = self.in_layer(x)
        out = self.mid_layer(out)
        out = self.out_layer(out)
        return out + shortcut

@MODELS.register_module()
class ASAT(nn.Module):
    def __init__(
        self, 
        in_channels, 
        reduced_ratio=4,
        patch_size=16,
        attn_repeats=2,
        attn_heads=4, 
        groups=4, 
        dropout=None,
        num_blocks=3
    ):
        super(ASAT, self).__init__()
        reduced_dim = in_channels // reduced_ratio
        self.groups = groups
        
        self.group_encoder = GroupEncoder(
            in_channels=in_channels,
            out_channels=reduced_dim,
            groups=groups
        )

        # self.attn = GroupSelfAttention(
        #     reduced_dim,
        #     attn_heads=attn_heads,
        #     repeats=attn_repeats,
        #     dropout=dropout
        # )
        
        # self.weighted_blocks = nn.ModuleList([
        #     nn.Sequential(
        #         nn.Conv2d(reduced_dim, reduced_dim, kernel_size=1),
        #         nn.BatchNorm2d(reduced_dim),
        #         nn.ReLU(inplace=True),
        #         nn.Dropout(dropout) if dropout is not None else nn.Identity(),
        #     ) for _ in range(num_blocks)
        # ])
        # self.output_layer = nn.Sequential(
        #     nn.Conv2d(reduced_dim, reduced_dim, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(reduced_dim),
        #     nn.ReLU(inplace=True),
        # )
        
    def forward(self, feat):
        feat = feat[0]
        out = self.group_encoder(feat) # [B, 512, 16, 16] * groups
        # out = self.attn(out) # [B, 512, 16, 16]
        
        # for block in self.weighted_blocks:
        #     out = block(out) + out
        # out = self.output_layer(out)
        
        return (out,)

