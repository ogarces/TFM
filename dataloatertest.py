
import os
import random
import warnings

import mmcv
import numpy as np
import torch
import torch.distributed as dist

from mmcv.utils import build_from_cfg

from mmsegmentation.mmseg.datasets import PotsdamDataset


import torch
import mmcv

import mmcv
from mmcv.utils import print_log

from mmseg.utils import get_root_logger
from mmseg.datasets import DATASETS
from mmseg.datasets import CustomDataset

import numpy as np




# class ImageDatasetP(CustomDataset):
#     def __init__(self, pipeline, data_root, split, file_client_args=dict(backend='disk')):

        
#         # Transforms for low resolution images and high resolution images
#         self.custom_classes = True
#         self.file_client_args = file_client_args
#         self.file_client = mmcv.FileClient.infer_client(self.file_client_args)
#         self.pipeline = pipeline
#         self.split = split
#         self.data_root = data_root
#         self.img_dir = os.path.join(self.data_root, 'img_dir', self.split)
#         self.ann_dir = os.path.join(self.data_root, 'ann_dir', self.split)
#         self.img_infos = self.load_annotations(img_dir=self.img_dir, img_suffix='png', seg_map_suffix='png',  ann_dir=self.ann_dir, split=None)
#         self.get_class
#     def __getitem__(self, index):
#         image = self.prepare_train_img(index)['img']
#         mask = self.prepare_train_img(index)['gt_semantic_seg']
        
#         print(mask)

#         if self.split == 'train':
            
#             one_hot = torch.nn.functional.one_hot(torch.Tensor(lab_hr).long(), num_classes=6)
#             # if np.min(lab_hr) == 2:
#             #     print( 'min', np.min(lab_hr), 'max', np.max(lab_hr), 'uniq', np.unique(lab_hr), 'one hot', one_hot)
#             #print('ONE HOT SIZE', one_hot, one_hot.shape)
#             lab_hr = np.transpose(torch.squeeze(one_hot), (2,0,1))
#             #print(one_hot.shape, img_hr.shape)
#             # print(np.unique(torch.argmax(one_hot, dim=1)))

#         return {"hr": image, 'lab_hr':mask}
        
        
#     def __len__(self):
#         return len(self.img_infos)
    
    

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

class Mydataset(CustomDataset):
   
    CLASSES = ('impervious_surface', 'building', 'low_vegetation', 'tree',
               'car', 'clutter')


    PALETTE = [[255, 255, 255], [0, 0, 255], [0, 255, 255], [0, 255, 0],
               [255, 255, 0], [255, 0, 0]]



    def __init__(self, **kwargs):
        super(Mydataset, self).__init__(
            img_suffix='.png',
            seg_map_suffix='.png',
            reduce_zero_label=True,
            **kwargs)

        assert self.file_client.exists(self.img_dir)

    def load_annotations(self,
                         img_dir,
                         img_suffix,
                         ann_dir,
                         seg_map_suffix=None,
                         split=None):
        

        img_infos = []
        if split is not None:
            with open(split) as f:
                for line in f:
                    name = line.strip()
                    img_info = dict(filename=name + img_suffix)
                    if ann_dir is not None:
                        ann_name = name 
                        seg_map = ann_name + seg_map_suffix
                        img_info['ann'] = dict(seg_map=seg_map)
                    img_infos.append(img_info)
        else:
            for img in mmcv.scandir(img_dir, img_suffix, recursive=True):
                img_info = dict(filename=img)
                if ann_dir is not None:
                    seg_img = img
                    seg_map = seg_img.replace(
                        img_suffix,  seg_map_suffix)
                    img_info['ann'] = dict(seg_map=seg_map)
                img_infos.append(img_info)
        return img_infos        
                
    def __getitem__(self, index):
        self.load_annotations(img_dir=img_dir, img_suffix='png', seg_map_suffix='png',  
                              ann_dir=ann_dir, split=None,)
    
        
        if 'train' in img_dir:
            train_data = self.prepare_train_img(index)
            image = train_data['img']
            mask = train_data['gt_semantic_seg']
            
                
                        
            one_hot = torch.nn.functional.one_hot(torch.Tensor(mask).long(), num_classes=6)
            # if np.min(lab_hr) == 2:
            #     print( 'min', np.min(lab_hr), 'max', np.max(lab_hr), 'uniq', np.unique(lab_hr), 'one hot', one_hot)
            #print('ONE HOT SIZE', one_hot, one_hot.shape)
            mask = np.transpose(torch.squeeze(one_hot), (2,0,1))
            print(one_hot.shape, mask.shape)
            print(np.unique(torch.argmax(one_hot, dim=1)))
            return train_data
            return {"hr": image, 'lab_hr':mask, 'meta': train_data['img_metas']}

        if 'val' in img_dir:
            
            train_data = self.prepare_train_img(index)
            image = train_data['img']
            mask = train_data['gt_semantic_seg']
            return train_data
            return {"hr": image, 'lab_hr':mask, 'meta': train_data['img_metas']}


        
    
    







from torch.utils.data import Dataset, DataLoader

class Mydataset1(CustomDataset):
   
    CLASSES = ('impervious_surface', 'building', 'low_vegetation', 'tree',
               'car', 'clutter')


    PALETTE = [[255, 255, 255], [0, 0, 255], [0, 255, 255], [0, 255, 0],
               [255, 255, 0], [255, 0, 0]]



    def __init__(self, pipeline, img_dir, ann_dir, split):
        


        self.split = split
        self.pipeline = pipeline
        self.img_dir = img_dir
        self.ann_dir = ann_dir
        self.img_infos = []
        
    def load_annotations(self,
                        img_dir,
                        img_suffix,
                        ann_dir,
                        seg_map_suffix=None,
                        split=None):
        

        img_infos = []
        if split is not None:
            with open(split) as f:
                for line in f:
                    name = line.strip()
                    img_info = dict(filename=name + img_suffix)
                    if ann_dir is not None:
                        ann_name = name 
                        seg_map = ann_name + seg_map_suffix
                        img_info['ann'] = dict(seg_map=seg_map)
                    img_infos.append(img_info)
        else:
            for img in mmcv.scandir(img_dir, img_suffix, recursive=True):
                img_info = dict(filename=img)
                if ann_dir is not None:
                    seg_img = img
                    seg_map = seg_img.replace(
                        img_suffix,  seg_map_suffix)
                    img_info['ann'] = dict(seg_map=seg_map)
                img_infos.append(img_info)
        return img_infos        
                
            

 
    def __getitem__(self, index):
        # self.load_annotations(img_dir=img_dir, img_suffix='png', seg_map_suffix='png',  
        #                       ann_dir=ann_dir, split=None,)
    
        
        if 'train' in img_dir:
            train_data = self.prepare_train_img(index)
            image = train_data['img']
            mask = train_data['gt_semantic_seg']
            
                
                        
            one_hot = torch.nn.functional.one_hot(torch.Tensor(mask).long(), num_classes=6)
            print(one_hot.shape)
            # if np.min(lab_hr) == 2:
            #     print( 'min', np.min(lab_hr), 'max', np.max(lab_hr), 'uniq', np.unique(lab_hr), 'one hot', one_hot)
            #print('ONE HOT SIZE', one_hot, one_hot.shape)
            mask = np.transpose(torch.squeeze(one_hot), (2,0,1))
            print(one_hot.shape, mask.shape)
            print(np.unique(torch.argmax(one_hot, dim=1)))
            return train_data
            return {"hr": image, 'lab_hr':mask, 'meta': train_data['img_metas']}

        if 'val' in img_dir:
            
            train_data = self.prepare_train_img(index)
            image = train_data['img']
            mask = train_data['gt_semantic_seg']
            return train_data
            return {"hr": image, 'lab_hr':mask, 'meta': train_data['img_metas']}
    def __len__(self):
        return len(self.img_infos)

        
    

img_dir='/share/ogarces/PRANC/mmsegmentation/data/potsdam/img_dir/train'
ann_dir='/share/ogarces/PRANC/mmsegmentation/data/potsdam/ann_dir/train'
img_dirv='/share/ogarces/PRANC/mmsegmentation/data/potsdam/img_dir/val'
ann_dirv='/share/ogarces/PRANC/mmsegmentation/data/potsdam/ann_dir/val'
data_root = '/share/ogarces/PRANC/mmsegmentation/data/potsdam'




# IMG_TRAIN_PATH = '/share/ogarces/PRANC/mmsegmentation/data/potsdam/img_dir/train'
# IMG_TEST_PATH = '/share/ogarces/PRANC/mmsegmentation/data/potsdam/img_dir/val'


# LAB_TRAIN_PATH = '/share/ogarces/PRANC/mmsegmentation/data/potsdam/ann_dir/train'
# LAB_TEST_PATH = '/share/ogarces/PRANC/mmsegmentation/data/potsdam/ann_dir/val'

# train_filenames = [os.path.join(IMG_TRAIN_PATH, o) for o in os.listdir(IMG_TRAIN_PATH)]
# train_masks = [os.path.join(LAB_TRAIN_PATH, o) for o in os.listdir(LAB_TRAIN_PATH)]
# test_filenames = [os.path.join(IMG_TEST_PATH, o) for o in os.listdir(IMG_TEST_PATH)]
# test_masks = [os.path.join(LAB_TEST_PATH, o) for o in os.listdir(LAB_TEST_PATH)]


trainset = Mydataset1(pipeline = train_pipeline, img_dir=img_dir, ann_dir=ann_dir,  split=None)    
trainset.load_annotations(img_dir=img_dir, img_suffix='png', seg_map_suffix='png',  ann_dir=ann_dir,
  
                        split=None,)

# trainset.reduce_zero_label

trainloader = torch.utils.data.DataLoader(trainset,  batch_size=12,  shuffle=False,
                                                drop_last = True, num_workers=2)
 
 
print(type(trainloader))   
# valset =  Mydataset(pipeline = val_pipeline, img_dir=img_dirv, ann_dir=ann_dirv,) 
# for i, j in enumerate(trainloader):
#     print(j)

for i in trainset:
    print(i)


# print(type(trainset))
# # print(trainset.img_infos)
# print(type(trainloader))
# # print(type(CustomDataset))
# print(trainset)


bset = CustomDataset(pipeline = train_pipeline, img_dir=img_dir, ann_dir=ann_dir, img_suffix='.png', )
bset.load_annotations(img_dir=img_dir, img_suffix='.png', ann_dir=ann_dir, seg_map_suffix='png', split=None)
bset.CLASSES =     CLASSES = ('impervious_surface', 'building', 'low_vegetation', 'tree',
               'car', 'clutter')

bset.PALETTE =  [[255, 255, 255], [0, 0, 255], [0, 255, 255], [0, 255, 0],
               [255, 255, 0], [255, 0, 0]]
print(type(bset))
print(bset.CLASSES)

bset.reduce_zero_label=True
print(bset.get_classes_and_palette())







class PotsdamDataset(CustomDataset):
    """ISPRS Potsdam dataset.

    In segmentation map annotation for Potsdam dataset, 0 is the ignore index.
    ``reduce_zero_label`` should be set to True. The ``img_suffix`` and
    ``seg_map_suffix`` are both fixed to '.png'.
    """
    CLASSES = ('impervious_surface', 'building', 'low_vegetation', 'tree',
               'car', 'clutter')

    PALETTE = [[255, 255, 255], [0, 0, 255], [0, 255, 255], [0, 255, 0],
               [255, 255, 0], [255, 0, 0]]

    def __init__(self, **kwargs):
        super(PotsdamDataset, self).__init__(
            img_suffix='.png',
            seg_map_suffix='.png',
            reduce_zero_label=True,
            **kwargs)
        
bset = PotsdamDataset(pipeline = train_pipeline, img_dir=img_dir, ann_dir=ann_dir,  split=None)

# bset.load_annotations(img_dir=img_dir, img_suffix='.png', ann_dir=ann_dir, seg_map_suffix='png', split=None)

print(type(bset))
