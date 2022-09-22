_base_ = [
    '../_base_/models/bts.py', 
    '../_base_/datasets/nyu.py',
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
        max_depth=10,
        loss_decode=dict(
            type='SigLoss', valid_mask=True, loss_weight=1.0)),
    )

data = dict(
    samples_per_gpu=8, # batchsize=16 on a dual-gpu node
    workers_per_gpu=8,
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
