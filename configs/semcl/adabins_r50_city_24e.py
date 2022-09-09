_base_ = [
    '../_base_/models/adabins.py', 
    '../_base_/datasets/cityscapes.py',
    '../_base_/default_runtime.py', 
    '../_base_/schedules/schedule_24x.py'
]

norm_cfg = dict(type='SyncBN', requires_grad=True)

model = dict(
    pretrained=None, # cannot directly use `https://dl.fbaipublicfiles.com/moco-v3/r-50-1000ep/r-50-1000ep.pth.tar` since the base_encoder is not extracted. Do that via semcl2bkb.py
    backbone=dict(
        _delete_=True,
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3, 4),
        style='pytorch',
        norm_cfg=norm_cfg,
        # init_cfg=dict(
        #     type='Pretrained', checkpoint='torchvision://resnet50'),
        ),
    decode_head=dict(
        in_channels=[64, 256, 512, 1024, 2048],
        up_sample_channels=[128, 256, 512, 1024, 2048],
        channels=128, # last one
        min_depth=1e-3,
        max_depth=10,
        norm_cfg=norm_cfg),
    )

data = dict(
    samples_per_gpu=8, # batchsize=16 on a dual-gpu node
    workers_per_gpu=2,
)

find_unused_parameters=True
SyncBN=True

# runtime
evaluation = dict(interval=1)

# use dynamicscale, and initialize with 512. 
# [已有模型 AMP 使用方法](https://zhuanlan.zhihu.com/p/375224982)
fp16 = dict(loss_scale=dict(init_scale=512.,mode='dynamic'))  