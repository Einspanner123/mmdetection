_base_ = [
    'mmdet::_base_/datasets/coco_algae.py',
    'mmdet::_base_/schedules/schedule_1x.py', 
    'mmdet::_base_/default_runtime.py'
]
custom_imports = dict(
    imports=['model', 'hooks.merged_hooks'],
    allow_failed_imports=False
)

model = dict(
    type='YOLOF',
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[103.530, 116.280, 123.675],
        std=[1.0, 1.0, 1.0],
        bgr_to_rgb=False,
        pad_size_divisor=32),
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(3, ),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=False),
        norm_eval=True,
        style='caffe',
        init_cfg=dict(
            type='Pretrained',
            checkpoint='open-mmlab://detectron/resnet50_caffe')),
    neck=dict(
        type='ASAT',
        in_channels=2048,
        reduced_ratio=4,
        attn_repeats=2,
        attn_heads=4,
        groups=4,
        dropout=0.2),
    bbox_head=dict(
        type='YOLOFHead',
        num_classes=10,
        in_channels=512,
        reg_decoded_bbox=True,
        anchor_generator=dict(
            type='AnchorGenerator',
            ratios=[1.0],
            scales=[1, 2, 4, 8, 16],
            strides=[32]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[1., 1., 1., 1.],
            add_ctr_clamp=True,
            ctr_clamp=32),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='GIoULoss', loss_weight=1.0)),
    # training and testing settings
    train_cfg=dict(
        assigner=dict(
            type='UniformAssigner', pos_ignore_thr=0.15, neg_ignore_thr=0.7),
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(
            type='soft_nms',
            iou_threshold=0.3,
            sigma=0.5,
            min_score=1e-3,
            method='gaussian'),
        max_per_img=100))
# optimizer
optim_wrapper = dict(
    optimizer=dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001),
    paramwise_cfg=dict(
        norm_decay_mult=0., custom_keys={'backbone': dict(lr_mult=1. / 3)}),
        clip_grad=dict(max_norm=35, norm_type=2)
    )

# learning rate
# param_scheduler = [
#     dict(
#         type='LinearLR',
#         start_factor=0.0001,
#         by_epoch=False,
#         begin=0,
#         end=2000),
#     dict(
#         type='MultiStepLR',
#         begin=0,
#         end=12,
#         by_epoch=True,
#         milestones=[8, 11],
#         gamma=0.1)
# ]
max_epochs = 300
param_scheduler = [
    dict(
        type='LinearLR', 
        start_factor=0.1,
        by_epoch=False, 
        begin=0, end=917),
    dict(
        type='CosineAnnealingLR',
        eta_min=0.0,
        begin=1,
        T_max=299,
        end=500,
        by_epoch=True,
        convert_to_iter_based=True)
]
train_cfg = dict(max_epochs=max_epochs, val_interval=1)

train_pipeline = [
    dict(type='LoadImageFromFile', backend_args={{_base_.backend_args}}),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='RandomShift', prob=0.5, max_shift_px=32),
    dict(type='PackDetInputs')
]
test_pipeline = [
    dict(type='LoadImageFromFile', backend_args={{_base_.backend_args}}),
    dict(type='Resize', scale=(1333, 800), keep_ratio=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]

dataset_type = 'CocoDataset'
evalute_type = 'CocoMetric'
batch_size = 8
train_dataloader = dict(
    batch_size=batch_size,
    num_workers=8,
    dataset=dict(type=dataset_type, pipeline=train_pipeline))
val_dataloader = dict(dataset=dict(pipeline=test_pipeline))
test_dataloader = val_dataloader
val_evaluator = dict(type=evalute_type)
test_evaluator = val_evaluator
vis_backends = [
    dict(type='LocalVisBackend'),
    dict(type='TensorboardVisBackend')]
visualizer = dict(
    type='DetLocalVisualizer', vis_backends=vis_backends, name='visualizer')

default_hooks = dict(checkpoint=dict(type='CheckpointHook', interval=1,
        save_best='coco/bbox_mAP', rule='greater', max_keep_ckpts=5))
custom_hooks = [
    dict(
        type='EMAHook',
        ema_type='ExpMomentumEMA',
        momentum=0.0002,
        update_buffers=True,
        priority=49),
    dict(
        type='EarlyStoppingHook',
        priority=50,
        patience=10,
        min_delta=0.001,
        monitor='coco/bbox_mAP',
        rule='greater'),
    dict(
        type='FeatureVisualizationHook',
        output_dir='./work_dirs/{{fileBasenameNoExtension}}/features_backbone',
        interval=100,
        feat_from='backbone',
        phase='test'),
    dict(
        type='FeatureVisualizationHook',
        output_dir='./work_dirs/{{fileBasenameNoExtension}}/features_neck',
        interval=100,
        feat_from='neck',
        phase='test'),
]
# NOTE: `auto_scale_lr` is for automatically scaling LR,
# USER SHOULD NOT CHANGE ITS VALUES.
# base_batch_size = (8 GPUs) x (8 samples per GPU)
auto_scale_lr = dict(base_batch_size=64)
