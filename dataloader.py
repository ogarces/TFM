from cgi import test
import os
import torch
import torchvision
from torchvision import datasets, models
from torchvision.transforms import transforms, Lambda
import collections
import json
#import cv2
import numpy as np
import albumentations as A
import cv2
#import scipy.misc as m
#import scipy.io as io
#import matplotlib.pyplot as plt
#import glob


from PIL import Image
from tqdm import tqdm
from torch.utils import data
# from skimage import io, transform
# from skimage import img_as_bool
import random
#from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torchvision.transforms.functional as TF
from torch.utils.data.sampler import SubsetRandomSampler
from mmseg.datasets import CustomDataset
import mmcv
#import matplotlib.pyplot as plt
#import matplotlib.image as mpimg



def DataLoader(args): 
    args.batch_size //= args.world_size
    if args.task == 'mnist':
        trainset = datasets.MNIST(args.dataset, download=True, train=True,transform=transforms.Compose( [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]))
        train_sampler = torch.utils.data.distributed.DistributedSampler(trainset)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=False, sampler = train_sampler, pin_memory=True)
        
        testset = datasets.MNIST(args.dataset, download=True, train=False,transform=transforms.Compose( [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]))
        test_sampler = torch.utils.data.distributed.DistributedSampler(testset, shuffle=False, drop_last=True)
        testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size,  shuffle=False, sampler=test_sampler, pin_memory=True)
        
        return trainloader, testloader

    if args.task == 'cifar10':
        transform_train = transforms.Compose([
            transforms.RandomCrop(args.img_width, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        trainset = torchvision.datasets.CIFAR10(root=args.dataset, train=True, download=True, transform=transform_train)
        train_sampler = torch.utils.data.distributed.DistributedSampler(trainset)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_worker, sampler=train_sampler, pin_memory=True)

        testset = torchvision.datasets.CIFAR10(root=args.dataset, train=False, download=True, transform=transform_test)
        test_sampler = torch.utils.data.distributed.DistributedSampler(testset, shuffle=False, drop_last=True)
        testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_worker, sampler=test_sampler, pin_memory=True)
        return trainloader, testloader

    if args.task == 'cifar100':
        transform_train = transforms.Compose([
            transforms.RandomCrop(args.img_width, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        trainset = torchvision.datasets.CIFAR100(root=args.dataset, train=True, download=True, transform=transform_train)
        train_sampler = torch.utils.data.distributed.DistributedSampler(trainset)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_worker, sampler=train_sampler, pin_memory=True)

        testset = torchvision.datasets.CIFAR100(root=args.dataset, train=False, download=True, transform=transform_test)
        test_sampler = torch.utils.data.distributed.DistributedSampler(testset, shuffle=False, drop_last=True)
        testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_worker, sampler=test_sampler, pin_memory=True)
        return trainloader, testloader

    if args.task == 'tiny':
        transform_train = transforms.Compose([
            transforms.RandomCrop(args.img_width, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        trainset = datasets.ImageFolder(os.path.join(args.dataset, "train"), transform=transform_train)
        train_sampler = torch.utils.data.distributed.DistributedSampler(trainset)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_worker, sampler=train_sampler, pin_memory=True)

        testset = datasets.ImageFolder(os.path.join(args.dataset, "val"), transform=transform_test)
        test_sampler = torch.utils.data.distributed.DistributedSampler(testset, shuffle=False, drop_last=True)
        testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_worker, sampler=test_sampler, pin_memory=True)
        return trainloader, testloader
    
    if args.task == 'imagenet' or args.task == 'imagenet100':
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])

        transform_test =  transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])
        trainset = datasets.ImageFolder(os.path.join(args.dataset, "train"), transform=transform_train)
        train_sampler = torch.utils.data.distributed.DistributedSampler(trainset)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_worker, sampler=train_sampler, pin_memory=True)

        testset = datasets.ImageFolder(os.path.join(args.dataset, "val"), transform=transform_test)
        test_sampler = torch.utils.data.distributed.DistributedSampler(testset, shuffle=False, drop_last=True)
        testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_worker, sampler=test_sampler, pin_memory=True)

        return trainloader, testloader
    
##############################################################################################    



    
    if args.task=='iSAID1':
        IMG_TRAIN_PATH = './datasets/iSAID/train/images/images1'
        IMG_TEST_PATH = './datasets/iSAID/val/images/images1'
        
        LAB_TRAIN_PATH = './datasets/iSAID/train/Semantic_masks/images1'
        LAB_TEST_PATH = './datasets/iSAID/val/Semantic_masks/images1'

        JSON_TRAIN_PATH = './datasets/iSAID/train/Annotations/data.json'
        JSON_TEST_PATH = './datasets/iSAID/val/Annotations/data.json'
 
        
        with open(JSON_TRAIN_PATH, 'rb') as f:
            train_json_dict = json.load(f)

        with open(JSON_TEST_PATH, 'rb') as f:
            test_json_dict = json.load(f)
                        
        train_filenames = []
        train_masks = []
        test_masks = []
        test_filenames = []
        for elem in train_json_dict['images']:
            train_filenames.append(IMG_TRAIN_PATH+'/'+elem['file_name'])
            train_masks.append(LAB_TRAIN_PATH+'/'+elem['seg_file_name'])
            
        for elem in test_json_dict['images']:
            test_filenames.append(IMG_TEST_PATH+'/'+elem['file_name'])
            test_masks.append(LAB_TEST_PATH+'/'+elem['seg_file_name'])

    
        print('number of train images', len(train_filenames))
        print(train_filenames)

        train_dataset = ImageDatasetN(train_filenames[:10], train_masks, hr_shape=(512 ,512))
        trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,
                                                shuffle=False,
                                                drop_last = True,num_workers=args.num_worker
                                                )
        # '/share/ogarces/PRANC/mmsegmentation/data/iSAID/img_dir/train/P2657_1553_2449_0_896.png']
        # './datasets/iSAID/train/images/images1/P1783_20482123.png', './datasets/iSAID/train/images/images1/P1783_23330.png']
        test_dataset = ImageDatasetN(test_filenames, test_masks, hr_shape=(512,512),
                                     )      
        testloader = torch.utils.data.DataLoader(test_dataset[:10],
                                                batch_size=args.batch_size,
                                                shuffle=False,  
                                                drop_last = True, num_workers=args.num_worker
                                                ) 
        return trainloader,testloader
    
   
    if args.task=='iSAID':
        IMG_TRAIN_PATH = '/share/ogarces/PRANC/mmsegmentation/data/iSAID/img_dir/train'
        IMG_TEST_PATH = '/share/ogarces/PRANC/mmsegmentation/data/iSAID/img_dir/val'


        LAB_TRAIN_PATH = '/share/ogarces/PRANC/mmsegmentation/data/iSAID/ann_dir/train'
        LAB_TEST_PATH = '/share/ogarces/PRANC/mmsegmentation/data/iSAID/ann_dir/val'
        
        train_filenames = [os.path.join(IMG_TRAIN_PATH, o) for o in os.listdir(IMG_TRAIN_PATH)]
        train_masks = [os.path.join(LAB_TRAIN_PATH, o) for o in os.listdir(LAB_TRAIN_PATH)]
        test_filenames = [os.path.join(IMG_TEST_PATH, o) for o in os.listdir(IMG_TEST_PATH)]
        test_masks = [os.path.join(LAB_TEST_PATH, o) for o in os.listdir(LAB_TEST_PATH)]


        #print(train_filenames)
        print('number of train images', len(train_filenames))

        train_dataset = ImageDatasetN(train_filenames, train_masks, hr_shape=(512 ,512))
        trainloader = torch.utils.data.DataLoader(train_dataset,
                                                batch_size=20,
                                        
                                                shuffle=False,
                                                drop_last = True, num_workers=args.num_worker
                                                )


        test_dataset = ImageDatasetN(test_filenames, test_masks, hr_shape=(512,512),
                                     )      
        testloader = torch.utils.data.DataLoader(test_dataset,
                                                batch_size=20,
                                                shuffle=False,  
                                                drop_last = True, num_workers=args.num_worker
                                                ) 
        return trainloader,testloader


    
    if args.task == 'potsdam':
        IMG_TRAIN_PATH = '/share/ogarces/PRANC/mmsegmentation/data/potsdam/img_dir/train'
        IMG_TEST_PATH = '/share/ogarces/PRANC/mmsegmentation/data/potsdam/img_dir/val'


        LAB_TRAIN_PATH = '/share/ogarces/PRANC/mmsegmentation/data/potsdam/ann_dir/train'
        LAB_TEST_PATH = '/share/ogarces/PRANC/mmsegmentation/data/potsdam/ann_dir/val'
        
        train_filenames = [os.path.join(IMG_TRAIN_PATH, o) for o in os.listdir(IMG_TRAIN_PATH)]
        train_masks = [os.path.join(LAB_TRAIN_PATH, o) for o in os.listdir(LAB_TRAIN_PATH)]
        test_filenames = [os.path.join(IMG_TEST_PATH, o) for o in os.listdir(IMG_TEST_PATH)]
        test_masks = [os.path.join(LAB_TEST_PATH, o) for o in os.listdir(LAB_TEST_PATH)]


        #print(train_filenames)
        print('number of train images', len(train_filenames))

        train_dataset = ImageDatasetP(train_filenames, train_masks, split = 'train'
                                      )
        print(type(train_dataset))
        trainloader = torch.utils.data.DataLoader(train_dataset,
                                                batch_size=12,
                                        
                                                shuffle=False,
                                                drop_last = True, num_workers=args.num_worker
                                                )

        print(trainloader)
        test_dataset = ImageDatasetP(test_filenames, test_masks, split = 'val'
                                     )      
        testloader = torch.utils.data.DataLoader(test_dataset,
                                                batch_size=12,
                                                shuffle=False,  
                                                drop_last = True, num_workers=args.num_worker
                                                ) 
        return trainloader,testloader


    if args.task == 'potsdam1':
        IMG_TRAIN_PATH = '/share/ogarces/PRANC/mmsegmentation/data/potsdam/img_dir/train'
        IMG_TEST_PATH = '/share/ogarces/PRANC/mmsegmentation/data/potsdam/img_dir/val'


        LAB_TRAIN_PATH = '/share/ogarces/PRANC/mmsegmentation/data/potsdam/ann_dir/train'
        LAB_TEST_PATH = '/share/ogarces/PRANC/mmsegmentation/data/potsdam/ann_dir/val'
        
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


        

        train_dataset = Mydataset(pipeline = train_pipeline, img_dir=IMG_TRAIN_PATH, ann_dir=LAB_TRAIN_PATH,)    
        print(type(train_dataset))
        train_dataset.load_annotations(img_dir=IMG_TRAIN_PATH, img_suffix='png', seg_map_suffix='png',  ann_dir=LAB_TRAIN_PATH,
                            split=None,)
        print(type(train_dataset))



        trainloader = torch.utils.data.DataLoader(train_dataset,  batch_size=8,  shuffle=True,
                                                        drop_last = True, num_workers=args.num_workers
                                                        )

        # <class 'dataloader.ImageDatasetP'>
        # <torch.utils.data.dataloader.DataLoader object at 0x7fba2f3df610>
        # will initiate now
        test_dataset = Mydataset(pipeline = val_pipeline, img_dir=IMG_TEST_PATH, ann_dir=LAB_TEST_PATH,)    
        test_dataset.load_annotations(img_dir=IMG_TEST_PATH, img_suffix='png', seg_map_suffix='png',  ann_dir=LAB_TEST_PATH,
                            split=None, )


        testloader = torch.utils.data.DataLoader(test_dataset,  batch_size=8,  shuffle=False,
                                                        drop_last = True, num_workers=args.num_workers
                                                        )
        return trainloader,testloader

    
    raise "Unknown task"





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
        

        # if img.mode != 'RGB':
        #     img = img.convert('RGB')
        
        # print(train_data.keys())
        
        if 'train' in self.img_dir:
            train_data = self.prepare_train_img(index)
            image = train_data['img'].data
            mask = train_data['gt_semantic_seg'].data
                
            #lab_hr = transforms.ToTensor()
            
            # print(lab_hr.shape, np.max(lab_hr), np.min(lab_hr))
            
            one_hot = torch.nn.functional.one_hot(torch.Tensor(mask).long(), num_classes=6)
            # if np.min(lab_hr) == 2:
            #     print( 'min', np.min(lab_hr), 'max', np.max(lab_hr), 'uniq', np.unique(lab_hr), 'one hot', one_hot)
            #print('ONE HOT SIZE', one_hot, one_hot.shape)
            lab_hr = np.transpose(torch.squeeze(one_hot), (2,0,1))



        if 'val' in self.img_dir:
            
            train_data = self.prepare_test_img(index)
            image = train_data['img'].data
            lab_hr = train_data['gt_semantic_seg'].data
        return {"hr": image, 'lab_hr':lab_hr, 'meta': train_data['img_metas']}











class ImageDatasetN(Dataset):
    def __init__(self, filename_list, filename_list_labs, hr_shape):
        hr_height, hr_width = hr_shape
        # Transforms for low resolution images and high resolution images

        self.hr_transform = transforms.Compose([transforms.ToTensor(),
                                                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                                                ])
        

        self.transform = A.Compose([ A.RandomCrop(height=hr_height, width=hr_height),])


                
        self.files = filename_list
        self.labs = filename_list_labs

    def __getitem__(self, index):
        image = Image.open(self.files[index % len(self.files)])
        mask = Image.open(self.labs[index % len(self.labs)])

        # if img.mode != 'RGB':
        #     img = img.convert('RGB')
        
        transformed = self.transform(image=np.array(image), mask=np.array(mask))
        img_hr = transformed['image']
        lab_hr = transformed['mask']
        img_hr = self.hr_transform(img_hr)
        lab_hr[lab_hr==255] = 0
        one_hot = torch.nn.functional.one_hot(torch.tensor(lab_hr).long(), num_classes=16)
        #print('after one hot', one_hot.shape)
        one_hot = np.transpose(torch.squeeze(one_hot), (2,0,1))
        #print(np.unique(torch.argmax(one_hot, dim=1)))
        
        return {"hr": img_hr, 'lab_hr':one_hot}
        
        
    def __len__(self):
        return len(self.files)
    
   
class ImageDatasetP(Dataset):
    def __init__(self, filename_list, filename_list_labs, split):
        
        # Transforms for low resolution images and high resolution images

        self.hr_transform = transforms.Compose([transforms.ToTensor(),
                                                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                                                ])
        
        # self.hr_transformLab = transforms.Compose([ transforms.ToTensor(), 
        #                                         ])
        self.transform = A.Compose([ A.HorizontalFlip(p=0.5),])

                    
        
        self.files = filename_list
        self.labs = filename_list_labs
        self.split = split

    def __getitem__(self, index):
        image = Image.open(self.files[index % len(self.files)])
        mask = Image.open(self.labs[index % len(self.labs)])

        # if img.mode != 'RGB':
        #     img = img.convert('RGB')
        
        
        transformed = self.transform(image=np.array(image), mask=np.array(mask))
        img_hr = transformed['image']
        lab_hr = transformed['mask']
        img_hr = self.hr_transform(img_hr)
        
        lab_hr[lab_hr == 0] = 255
        lab_hr = lab_hr - 1
        lab_hr[lab_hr == 254] = 255
        if self.split == 'train':
            
            #lab_hr = transforms.ToTensor()
            
            # print(lab_hr.shape, np.max(lab_hr), np.min(lab_hr))
            
            one_hot = torch.nn.functional.one_hot(torch.Tensor(lab_hr).long(), num_classes=7)
            # if np.min(lab_hr) == 2:
            #     print( 'min', np.min(lab_hr), 'max', np.max(lab_hr), 'uniq', np.unique(lab_hr), 'one hot', one_hot)
            #print('ONE HOT SIZE', one_hot, one_hot.shape)
            lab_hr = np.transpose(torch.squeeze(one_hot), (2,0,1))
            #print(one_hot.shape, img_hr.shape)
            # print(np.unique(torch.argmax(one_hot, dim=1)))

        return {"hr": img_hr, 'lab_hr':lab_hr}
        
        
    def __len__(self):
        return len(self.files)
    
    

