# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
from typing import Optional, Sequence

import cv2
import mmcv
import numpy as np
import torch
from mmengine.fileio import get
from mmengine.hooks import Hook
from mmengine.runner import Runner
from mmengine.utils import mkdir_or_exist
from mmengine.visualization import Visualizer

from mmdet.registry import HOOKS
from mmdet.structures import DetDataSample


@HOOKS.register_module()
class FPNFeatureVisualizationHook(Hook):
    """BiFPN Feature Visualization Hook. Used to visualize BiFPN features
    overlaid on original images during validation.

    Args:
        output_dir (str): Directory to save visualization results.
        interval (int): The interval of visualization. Defaults to 50.
        phase (str): Which phase to visualize. Choices are 'train' and 'val'.
            Defaults to 'val'.
        backend_args (dict, optional): Arguments to instantiate the
            corresponding backend. Defaults to None.
    """

    def __init__(self,
                 output_dir: str,
                 interval: int = 50,
                 phase: str = 'val',
                 backend_args: Optional[dict] = None):
        self.output_dir = output_dir
        self.interval = interval
        self.phase = phase
        self.backend_args = backend_args
        mkdir_or_exist(self.output_dir)
        self._visualizer: Visualizer = Visualizer.get_current_instance()

    def _process_features(self, runner: Runner, batch_idx: int,
                        outputs: Sequence[DetDataSample]) -> None:
        """Process and visualize BiFPN features.

        Args:
            runner (:obj:`Runner`): The runner of the process.
            batch_idx (int): The index of the current batch.
            outputs (Sequence[:obj:`DetDataSample`]]): A batch of data samples
                that contain annotations and predictions.
        """
        # Only visualize at specified intervals
        total_curr_iter = runner.iter + batch_idx
        if total_curr_iter % self.interval != 0:
            return

        # Get the model
        model = runner.model

        # Get the original image
        img_path = outputs[0].img_path
        img_bytes = get(img_path, backend_args=self.backend_args)
        img = mmcv.imfrombytes(img_bytes, channel_order='rgb')

        # Get the preprocessed image
        data_preprocessor = model.data_preprocessor
        device = next(model.parameters()).device
        
        # Create a batch with a single image
        batch_inputs = {}
        batch_inputs['inputs'] = [torch.from_numpy(img.transpose(2, 0, 1)).float().to(device)]
        batch_inputs = data_preprocessor(batch_inputs)
        
        # Extract features
        with torch.no_grad():
            x = model.backbone(batch_inputs['inputs'])
            bifpn_features = model.neck(x)

        # Visualize each feature level
        for i, feat in enumerate(bifpn_features):
            # Convert feature to numpy and normalize
            feat_np = feat.squeeze(0).mean(dim=0).cpu().numpy()
            feat_np = (feat_np - feat_np.min()) / (feat_np.max() - feat_np.min() + 1e-10)
            
            # Resize feature map to match image size
            feat_np = cv2.resize(feat_np, (img.shape[1], img.shape[0]))
            
            # Apply colormap to create heatmap
            heatmap = cv2.applyColorMap((feat_np * 255).astype(np.uint8), cv2.COLORMAP_JET)
            heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
            
            # Overlay heatmap on original image with transparency
            alpha = 0.5
            overlay = cv2.addWeighted(img, 1 - alpha, heatmap, alpha, 0)
            
            # Save the visualization
            img_name = osp.basename(img_path)
            # Add current epoch number as prefix to the filename
            epoch = runner.epoch + 1  # +1 because epoch is 0-indexed
            save_path = osp.join(self.output_dir, f"{epoch}_{osp.splitext(img_name)[0]}_P{i+3}.png")
            mmcv.imwrite(overlay, save_path)
            
    def _visualize_features_with_mmdet(self, runner: Runner, batch_idx: int,
                        outputs: Sequence[DetDataSample]) -> None:
        """Process and visualize BiFPN features using mmdetection's Visualizer.

        Args:
            runner (:obj:`Runner`): The runner of the process.
            batch_idx (int): The index of the current batch.
            outputs (Sequence[:obj:`DetDataSample`]]): A batch of data samples
                that contain annotations and predictions.
        """
        # Only visualize at specified intervals
        total_curr_iter = runner.iter + batch_idx
        if total_curr_iter % self.interval != 0:
            return

        # Get the model
        model = runner.model

        # Get the original image
        img_path = outputs[0].img_path
        img_bytes = get(img_path, backend_args=self.backend_args)
        img = mmcv.imfrombytes(img_bytes, channel_order='rgb')

        # Get the preprocessed image
        data_preprocessor = model.data_preprocessor
        device = next(model.parameters()).device
        
        # Create a batch with a single image
        batch_inputs = {}
        batch_inputs['inputs'] = [torch.from_numpy(img.transpose(2, 0, 1)).float().to(device)]
        batch_inputs = data_preprocessor(batch_inputs)
        
        # Extract features
        with torch.no_grad():
            x = model.backbone(batch_inputs['inputs'])
            bifpn_features = model.neck(x)

        # Visualize each feature level using the visualizer
        for i, feat in enumerate(bifpn_features):
            # 获取特征图原始尺寸
            _, _, feat_h, feat_w = feat.shape
            
            # 将输入图像缩放到特征图尺寸
            resized_img = cv2.resize(img, (feat_w, feat_h))
            
            # 转换并归一化特征图
            feat_np = feat.squeeze(0).mean(dim=0).cpu().numpy()
            feat_np = (feat_np - feat_np.min()) / (feat_np.max() - feat_np.min() + 1e-10)
            
            # 使用可视化器在缩放后的图像上绘制热力图
            drawn_img = resized_img.copy()
            self._visualizer.set_image(drawn_img)
            heatmap = torch.from_numpy(feat_np * 255).float().unsqueeze(0)
            self._visualizer.draw_featmap(heatmap, drawn_img, alpha=0.5)
            vis_img = self._visualizer.get_image()
            
            # 将结果还原到原始图像尺寸
            vis_img = cv2.resize(vis_img, (img.shape[1], img.shape[0]))
            
            # Save the visualization
            img_name = osp.basename(img_path)
            # Add current epoch number as prefix to the filename
            epoch = runner.epoch + 1  # +1 because epoch is 0-indexed
            save_path = osp.join(self.output_dir, f"{epoch}_{osp.splitext(img_name)[0]}_P{i+3}_mm.png")
            mmcv.imwrite(vis_img, save_path)

    def after_val_iter(self, runner: Runner, batch_idx: int, data_batch: dict,
                       outputs: Sequence[DetDataSample]) -> None:
        """Run after every ``self.interval`` validation iterations.

        Args:
            runner (:obj:`Runner`): The runner of the validation process.
            batch_idx (int): The index of the current batch in the val loop.
            data_batch (dict): Data from dataloader.
            outputs (Sequence[:obj:`DetDataSample`]]): A batch of data samples
                that contain annotations and predictions.
        """
        if self.phase != 'val':
            return
        self._process_features(runner, batch_idx, outputs)
        
    def after_test_iter(self, runner: Runner, batch_idx: int, data_batch: dict,
                        outputs: Sequence[DetDataSample]) -> None:
        """Run after every ``self.interval`` test iterations.

        Args:
            runner (:obj:`Runner`): The runner of the test process.
            batch_idx (int): The index of the current batch in the test loop.
            data_batch (dict): Data from dataloader.
            outputs (Sequence[:obj:`DetDataSample`]]): A batch of data samples
                that contain annotations and predictions.
        """
        if self.phase != 'test':
            return
        self._visualize_features_with_mmdet(runner, batch_idx, outputs)