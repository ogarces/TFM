# dataset settings
dataset_type = 'PotsdamDataset'
data_root = '/share/ogarces/PRANC/mmsegmentation/data/potsdam/'

# img_dir='/share/ogarces/PRANC/mmsegmentation/data/potsdam/img_dir/train'
# ann_dir='/share/ogarces/PRANC/mmsegmentation/data/potsdam/ann_dir/train'

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
crop_size = (512, 512)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=True),
    dict(type='Resize', img_scale=(512, 512), ratio_range=(0.5, 2.0)),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=5),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]

val_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(512, 512),
        # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=None,
        img_ratios=[1.0],
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=8,
    workers_per_gpu=8,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='img_dir/train',
        ann_dir='ann_dir/train',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='img_dir/val',
        ann_dir='ann_dir/val',
        pipeline=val_pipeline),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='img_dir/val_all',
        ann_dir='ann_dir/val_all',
        pipeline=test_pipeline))

print(data)


from mmseg.datasets import PotsdamDataset

img_dir='/share/ogarces/PRANC/mmsegmentation/data/potsdam/img_dir/train'
ann_dir='/share/ogarces/PRANC/mmsegmentation/data/potsdam/ann_dir/train'
img_dirv='/share/ogarces/PRANC/mmsegmentation/data/potsdam/img_dir/val'
ann_dirv='/share/ogarces/PRANC/mmsegmentation/data/potsdam/ann_dir/val'
data_root = '/share/ogarces/PRANC/mmsegmentation/data/potsdam'

trainset = PotsdamDataset(pipeline = train_pipeline, img_dir=img_dir, ann_dir=ann_dir)
trainset.load_annotations(img_dir=img_dir, img_suffix='png', seg_map_suffix='png',  ann_dir=ann_dir,
                        split=None,)

trainset.prepare_test_img

import torch
print(type(trainset))
trainloader = torch.utils.data.DataLoader(trainset,  batch_size=12,  shuffle=False,
                                                drop_last = True, num_workers=8)
print(trainset)
print(type(trainloader))
print(trainloader)
import os
from PIL import Image
for i, j in enumerate(trainloader):
    print(type(j))