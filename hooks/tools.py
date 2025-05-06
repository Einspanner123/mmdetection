import numpy as np
import torch
import matplotlib.pyplot as plt
from torchvision import transforms as T
import torch.nn.functional as F


def show_heatmap(image, feature_map, alpha=0.7, cmap='jet', title=None, save_path=None):
    """
    在原图上叠加特征图形成热力图并显示

    Args:
        image (PIL.Image | torch.Tensor): 原始图像，可以是PIL图像或张量
        feature_map (torch.Tensor): 特征图张量，形状为(1, c, h, w)
        alpha (float): 热力图的透明度，范围[0, 1]
        cmap (str): 热力图使用的颜色映射
        title (str): 图像标题
        save_path (str): 保存路径，如果为None则直接显示
    """

    # 确保feature_map是CPU张量
    feature_map = feature_map.detach().cpu()

    # 处理特征图：如果是多通道，取平均值
    if feature_map.ndim == 4:  # (1, c, h, w)
        if feature_map.size(1) > 1:  # 多通道
            feature_map = feature_map.mean(dim=1, keepdim=True)  # 通道平均
        feature_map = feature_map.squeeze(0)  # 移除批次维度，变为(c, h, w)或(1, h, w)

    # 确保特征图是单通道
    if feature_map.ndim == 3 and feature_map.size(0) > 1:
        feature_map = feature_map.mean(dim=0, keepdim=True)  # 如果还有多通道，取平均
    
    if feature_map.ndim == 3:
        feature_map = feature_map.squeeze(0)  # 变为(h, w)

    # 归一化特征图到[0, 1]范围
    if feature_map.min() < 0 or feature_map.max() > 1:
        feature_map = (feature_map - feature_map.min()) / (feature_map.max() - feature_map.min() + 1e-8)

    # 处理输入图像
    if isinstance(image, torch.Tensor):
        # 确保图像是CPU张量
        image = image.detach().cpu()
        
        # 处理图像形状
        if image.ndim == 4 and image.size(0) == 1:  # (1, c, h, w)
            image = image.squeeze(0)  # 移除批次维度，变为(c, h, w)
        
        # 转换为PIL图像以便后续处理
        if image.ndim == 3 and image.size(0) in {1, 3}:  # (c, h, w)
            # 反归一化（如果需要）
            # from dataset import HP
            # for t, m, s in zip(image, HP["MEAN"], HP["STD"]):
            #     t.mul_(s).add_(m)
            
            # 转换为PIL图像
            image = T.ToPILImage()(image)
    
    # 将特征图调整为与原图相同大小
    orig_size = image.size[::-1]  # PIL图像的size是(width, height)，需要转为(height, width)
    feature_map_np = feature_map.numpy()
    feature_map_resized = np.array(
        T.Resize(orig_size, interpolation=T.InterpolationMode.BICUBIC)(
            T.ToPILImage()(torch.tensor(feature_map_np).unsqueeze(0))
            )
        )
    
    # 创建图像
    plt.figure(figsize=(10, 8))
    
    # 显示原图
    plt.imshow(image)
    
    # 叠加热力图
    plt.imshow(feature_map_resized, alpha=alpha, cmap=cmap)
    
    # 添加颜色条
    plt.colorbar(label='Feature Intensity')
    
    # 设置标题
    if title:
        plt.title(title)
    else:
        plt.title('Feature Map Heatmap')
    
    plt.axis('off')
    plt.tight_layout()
    
    # 保存或显示
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()