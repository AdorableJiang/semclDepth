_base_ = [
    '../_base_/models/bts.py', 
    '../_base_/datasets/cityscapes_semcl.py',
    '../_base_/default_runtime.py', 
    '../_base_/schedules/schedule_24x.py'
]

pretrained='../moco4semencontrast/pretrained/bkb_r-50-1000ep.pth.tar', # cannot directly use `https://dl.fbaipublicfiles.com/moco-v3/r-50-1000ep/r-50-1000ep.pth.tar` since the base_encoder is not extracted. Do that via semcl2bkb.py
model = dict(
    backbone=dict(
        init_cfg=dict(
            type='Pretrained', checkpoint=pretrained),
        norm_cfg = dict(type='SyncBN', requires_grad=True)
    ),
    decode_head=dict(
        final_norm=False,
        min_depth=1e-3,
        max_depth=200, # this is dataset decided, use 200 as in https://github.com/zhyever/Monocular-Depth-Estimation-Toolbox/blob/9e490dda2940480cbedbf2893065b1e040691677/configs/_base_/datasets/cityscapes.py#L51
        loss_decode=dict(
            type='SigLoss', valid_mask=True, loss_weight=1.0)),
    )
    
dataset_type = 'CSsemclDataset'
data_root = 'data/cityscapes'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='DisparityLoadAnnotations'),
    dict(type='Resize', img_scale=(1216, 352), keep_ratio=False),
    dict(type='KBCrop', depth=True),
    dict(type='RandomRotate', prob=0.5, degree=2.5),
    dict(type='RandomFlip', prob=0.5),
    dict(type='RandomCrop', crop_size=(352, 704)),
    dict(type='ColorAug', prob=1, gamma_range=[0.9, 1.1], brightness_range=[0.9, 1.1], color_range=[0.9, 1.1]),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', 
         keys=['img', 'depth_gt'], 
         meta_keys=('filename', 'ori_filename', 'ori_shape',
                    'img_shape', 'pad_shape', 'scale_factor', 
                    'flip', 'flip_direction', 'img_norm_cfg',
                    'cam_intrinsic')),
]
data = dict(
    samples_per_gpu=8, # batchsize=16 on a dual-gpu node
    workers_per_gpu=8,
    train=dict(
        type='RepeatDataset',
        times=8,
        dataset=dict(
            type=dataset_type,
            data_root=data_root,
            img_dir='leftImg8bit',
            cam_dir='camera',
            ann_dir='disparity',
            depth_scale=256,
            split='cityscapes_train.txt',
            pipeline=train_pipeline,
            garg_crop=True,
            eigen_crop=False,
            min_depth=1e-3,
            max_depth=200
        )
    )
)

# optimizer
max_lr=1e-4
optimizer = dict(type='AdamW', lr=max_lr, betas=(0.95, 0.99), weight_decay=0.01,)
# learning policy
lr_config = dict(
    policy='OneCycle',
    max_lr=max_lr,
    div_factor=25,
    final_div_factor=100,
    by_epoch=False,
)
checkpoint_config = dict(by_epoch=True, max_keep_ckpts=2)
# runtime
evaluation = dict(
    interval=1,
    rule='less', 
    save_best='abs_rel',
    greater_keys=("a1", "a2", "a3"), 
    less_keys=("abs_rel", "rmse")
)

# use dynamicscale, and initialize with 512. 
# [已有模型 AMP 使用方法](https://zhuanlan.zhihu.com/p/375224982)
fp16 = dict(loss_scale='dynamic')
