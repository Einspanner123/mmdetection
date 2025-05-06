# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
from typing import Optional, Sequence

import cv2
import mmcv
import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F
from mmengine.fileio import get
from mmengine.hooks import Hook
from mmengine.runner import Runner
from mmengine.utils import mkdir_or_exist
from mmengine.visualization import Visualizer

from mmdet.registry import HOOKS
from mmdet.structures import DetDataSample


@HOOKS.register_module()
class FeatureVisualizationHook(Hook):
    """Merged BiFPN Feature Visualization Hook. Used to visualize BiFPN features
    overlaid on original images during validation.

    Args:
        output_dir (str): Directory to save visualization results.
        interval (int): The interval of visualization. Defaults to 50.
        phase (str): Which phase to visualize. Choices are 'train' and 'val'.
            Defaults to 'val'.
        vis_method (str): Visualization method to use. Choices are 'cv2' and 'mmdet'.
            Defaults to 'cv2'.
        backend_args (dict, optional): Arguments to instantiate the
            corresponding backend. Defaults to None.
    """

    def __init__(self,
                 output_dir: str,
                 interval: int = 50,
                 feat_from: str = 'neck',
                 phase: str = 'val',
                 vis_method: str = 'cv2',
                 backend_args: Optional[dict] = None):
        self.output_dir = output_dir
        self.interval = interval
        assert feat_from in ['neck','backbone']
        self.feat_from = feat_from
        assert phase in ['train', 'val', 'test']
        self.phase = phase
        assert vis_method in ['cv2', 'mmdet']
        self.vis_method = vis_method
        self.backend_args = backend_args
        mkdir_or_exist(self.output_dir)
        self._visualizer: Visualizer = Visualizer.get_current_instance()

    def _visualize_features(self, runner: Runner, batch_idx: int,
                        outputs: Sequence[DetDataSample]) -> None:
        # Only visualize at specified intervals
        total_curr_iter = runner.iter + batch_idx
        if total_curr_iter % self.interval != 0:
            return

        # Get the model
        model = runner.model

        # Get the original image
        img_path = outputs[0].img_path
        img_shape = outputs[0].metainfo['img_shape']
        img_bytes = get(img_path, backend_args=self.backend_args)
        img_name = osp.basename(img_path)
        # Add current epoch number as prefix to the filename
        epoch = runner.epoch + 1  # +1 because epoch is 0-indexed
        img = mmcv.imfrombytes(img_bytes, channel_order='rgb') # H, W, 3
        # Get the preprocessed image
        data_preprocessor = model.data_preprocessor
        device = next(model.parameters()).device
        
        # Create a batch with a single image
        batch_inputs = {}
        batch_inputs['inputs'] = [torch.from_numpy(img.transpose(2, 0, 1)).float().to(device)]
        batch_inputs = data_preprocessor(batch_inputs)
        # Extract features
        with torch.no_grad():
            features = model.backbone(batch_inputs['inputs'])
            if self.feat_from == 'neck':
                features = model.neck(features)
        
        image = batch_inputs['inputs'][0].permute(1, 2, 0).cpu().numpy()
        if image.min() < 0 or image.max() > 1:
            image = (image - image.min()) / (image.max() - image.min())
        H, W, _ = image.shape
        
        """ Visualize Heat Map """
        for i, feat in enumerate(features):
            feat = feat.mean(dim=1, keepdim=True).cpu() # 1, 1, H, W
            feat = F.interpolate(feat, size=(H, W), mode='bilinear', align_corners=False)
            feat_np = feat.squeeze(0).permute(1, 2, 0).numpy()
            if feat_np.min() < 0 or feat_np.max() > 1:
                feat_np = (feat_np - feat_np.min()) / (feat_np.max() - feat_np.min())
            
            plt.figure(figsize=(10, 8))
            plt.imshow(image)
            plt.imshow(feat_np, alpha=0.6, cmap='jet')
            plt.colorbar(label='Feature Intensity')
            plt.title('Feature Map Heatmap')
            plt.axis('off')
            plt.tight_layout()
            
            save_path = osp.join(
                self.output_dir,
                f"epoch{epoch}_{osp.splitext(img_name)[0]}_{self.feat_from}_P{i+3}.png"
            )
            plt.savefig(save_path)
            plt.close()


    def after_val_iter(
        self, 
        runner: Runner, 
        batch_idx: int, 
        data_batch: dict,
        outputs: Sequence[DetDataSample]) -> None:

        if self.phase == 'val':
            self._visualize_features(runner, batch_idx, outputs)
        
    def after_test_iter(
        self, 
        runner: Runner, 
        batch_idx: int, 
        data_batch: dict,
        outputs: Sequence[DetDataSample]) -> None:

        if self.phase == 'test':
            self._visualize_features(runner, batch_idx, outputs)
