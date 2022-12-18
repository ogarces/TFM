#%%
import cv2
import matplotlib.pyplot as plt
import os
import torch
import os
from os import listdir

ima = "/share/ogarces/PRANC/datasets/iSAID/train/Semantic_masks/images1/P2759_07799_instance_color_RGB.png"
imar = cv2.imread(ima)
print(imar[0,0,:])
#print(imar[0].shape)
imar[0,0,:] = [1,1,1]
print(imar[0,0,:])

t1 = torch.tensor([2,3,4])
t2 = torch.tensor([1,1,3])

print(t1*t2)
# import torch


# # %%
# model = torch.jit.load('/share/ogarces/PRANC/SEGMENTResnet50_iSAID_NORM/best_modelnormal2.pt')
# model.eval()
# print(model('/share/ogarces/PRANC/datasets/iSAID/val/images/images1/P0004_00.png'))

# # %%

# from utils import load_model


# %%
import models
import torch
from matplotlib import pyplot as plt
from torchvision import transforms
from PIL import Image
from torchvision import transforms, utils

model = models.deeplabv3(7)

# original saved file with DataParallel
checkpoint = torch.load('/share/ogarces/PRANC/SEGMENTResnet50_iSAID_NORM/best_modelnormal9Dpotsdam.pt', 
                        map_location=torch.device('cpu'))

model.load_state_dict(checkpoint, strict=False)
model.eval()


hr_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

im = Image.open('/share/ogarces/PRANC/mmsegmentation/data/potsdam/img_dir/val/4_13_1024_4608_1536_5120.png')          
mask = Image.open('/share/ogarces/PRANC/mmsegmentation/data/potsdam/ann_dir/val/4_13_1024_4608_1536_5120.png')                              
                    
timag = hr_transform(im)
print(im.size, timag.shape)


timag =  torch.unsqueeze(timag, dim=0)
#out = model(timag)
out = model(timag)['out'] # deeplab

out0 = torch.argmax(out, dim=1)
print('out, image shape', out0.shape, timag.shape)
plt.imshow(torch.squeeze(out0))
plt.show()

plt.imshow(im)
plt.show()
print(im)

plt.imshow(mask)
plt.show()


data_dir = '/share/ogarces/PRANC/datasets/iSAID/test/images/'
test_transforms = transforms.Compose([transforms.Resize(224),
                                      transforms.ToTensor(),
                                     ])

import dataloader
import numpy as np

model.eval()
print(out0.shape, timag.shape, np.array(mask).shape)




# %%
# import pickle
# import torch
# from matplotlib import pyplot as plt

# with open(r"./IMAG.pickle", "rb") as input_file:
#     im = pickle.load(input_file)


# with open(r"./outputs.pickle", "rb") as input_file:
#     o = pickle.load(input_file)
    
# with open(r"./labs.pickle", "rb") as input_file:
#     l = pickle.load(input_file)

# #print(im[0].cpu().shape)
# for i in range(len(im)):
#     print(i)
#     plt.imshow(torch.argmax(im[i].cpu(), dim=0))
#     plt.show()

#     plt.imshow(torch.argmax(o[i].cpu(), dim=0))
#     plt.show()

#     plt.imshow(torch.argmax(l[i].cpu(), dim=0))
#     plt.show()


 # %%
# # # # import numpy as np
# # # # rgb_masks0 = []
# # # # for e in listdir('/share/ogarces/PRANC/datasets/iSAID/train/Semantic_masks/images1/')[:1]:
# # # #     im1 = cv2.imread(os.path.join('/share/ogarces/PRANC/datasets/iSAID/train/Semantic_masks/images1/', e))
# # # #     #plt.imshow(im1)
        
# # # #     #plt.show()
# # # #     #print(np.max(im1), np.min(im1))

# # # #     for i in im1: 

    
# # # #         for n in i:
# # # #             if n.any() != 0 :
# # # #                 rgb_masks0.append(n)

# # # # rgb_masks0 = np.array(rgb_masks0)
# # # # rgb_masks0 = np.unique(rgb_masks0, axis=0)
# # # # print('transf', len(rgb_masks0))
# # # # print(rgb_masks0)    

# # # # # %%


# # # # # %%



# # # # # %%




# # #  # %%


    
    
    
# %%

# %%



# %%
# from matplotlib import pyplot as plt
# import torch
# from mmsegmentation import mmseg
# from mmsegmentation.mmseg.apis import inference_segmentor, init_segmentor
# from mmsegmentation.mmseg.datasets import iSAIDDataset



# import mmcv

# pipeline = mmseg.datasets.pipelines.MultiScaleFlipAug(img_scale=(256, 256),img_ratios=[0.5, 1.0],transforms=[
#                                             dict(type='Resize', keep_ratio=True),
#                                             dict(type='Pad', size_divisor=32),
#                                             dict(type='ImageToTensor', keys=['img']),
#                                             dict(type='Collect', keys=['img']),
# ])
# img_dir='/share/ogarces/PRANC/mmsegmentation/data/iSAID/img_dir/train'
# ann_dir='/share/ogarces/PRANC/mmsegmentation/data/iSAID/ann_dir/train'
# dataset = iSAIDDataset( pipeline=[pipeline], img_dir=img_dir, ann_dir=ann_dir)

# dataset.load_annotations(seg_map_suffix='_instance_color_RGB.png', 
#                          ann_dir=img_dir,
#                          img_dir=ann_dir,
#                          img_suffix='png',

#                            )
# import numpy as np
# print(np.unique(dataset.get_gt_seg_map_by_idx(3000)))
# print(dataset.get_gt_seg_map_by_idx(3000))
# print(dataset.get_gt_seg_map_by_idx(3000).shape)
# print(dataset.CLASSES)


# # ima = "/share/ogarces/PRANC/datasets/iSAID/train/Semantic_masks/images1/P2759_07799_instance_color_RGB.png"
# # imar = cv2.imread(ima)
# # print(imar.shape)




# # %%





# # # %%
# # import os
# # import cv2
# # from sklearn.preprocessing import OneHotEncoder
# # import numpy as np
# # from PIL import Image
# # masks_dir = '/share/ogarces/PRANC/mmsegmentation/data/iSAID/ann_dir/train/'
# # for i in os.listdir(masks_dir):
# #     path = os.path.join(masks_dir, i)
# #     im= cv2.imread(path)
# #     print(np.max(im))
# #     print(np.min(im))
# #     print(np.unique(im))
    
    

# %%


# %%
import torch
import segmentation_models_pytorch as smp

# lets assume we have multilabel prediction for 3 classes
output = torch.rand([10, 3, 256, 256])
target = torch.rand([10, 3, 256, 256])

# first compute statistics for true positives, false positives, false negative and
# true negative "pixels"
tp, fp, fn, tn = smp.metrics.get_stats(torch.argmax(output, dim=1), torch.argmax(target, dim=1), mode='multiclass', num_classes=3)

# then compute metrics with required reduction (see metric docs)
iou_score = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")
f1_score = smp.metrics.f1_score(tp, fp, fn, tn, reduction="micro")
f2_score = smp.metrics.fbeta_score(tp, fp, fn, tn, beta=2, reduction="micro")
accuracy = smp.metrics.accuracy(tp, fp, fn, tn, reduction="macro")
recall = smp.metrics.recall(tp, fp, fn, tn, reduction="micro-imagewise")



output2 = torch.rand([10, 3, 256, 256])
target2 = torch.rand([10, 3, 256, 256])

tp2, f2p, fn2, tn2 = smp.metrics.get_stats(torch.argmax(output2, dim=1), torch.argmax(target2, dim=1), mode='multiclass', num_classes=3)

print(tp2.shape)
# %%
print(torch.sum(tp, dim=0), torch.sum(tp, dim=1), target.shape, output.shape)

# %%
import torch
torch.cuda.empty_cache()



# %%
import albumentations as A 
from PIL import Image
import numpy as np
import cv2

transform = A.Compose([
    A.HorizontalFlip(p=0.5),
   
])

import torch
from PIL import Image
im = '/share/ogarces/PRANC/mmsegmentation/data/potsdam/ann_dir/val/4_13_1024_4608_1536_5120.png'
ma = '/share/ogarces/PRANC/mmsegmentation/data/iSAID/ann_dir/train/P0999_0_896_0_896_instance_color_RGB.png'
image = np.array(Image.open(im))
mask = np.array(Image.open(ma))
#print(mask[0])
m = (mask == 255).all(axis=0)
mask[mask==255] = 0
print(mask)

#print(np.array(mask).shape, mask, np.unique(mask))
# if img.mode != 'RGB':
#     img = img.convert('RGB')
#print(transform(image=image, mask=mask))
from matplotlib import pyplot as plt


# print(torch.squeeze(torch.unique(torch.Tensor(mask), dim=2)).shape)


# %%

    
# %%
hr_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

im = Image.open('/share/ogarces/PRANC/mmsegmentation/data/potsdam/img_dir/val/4_13_1024_4608_1536_5120.png')          
mask = Image.open('/share/ogarces/PRANC/mmsegmentation/data/potsdam/ann_dir/val/4_13_1024_4608_1536_5120.png')                              
                    
print(im.size)
print(im)
timag = torch.Tensor(im)
timag =  torch.unsqueeze(im, dim=1)
print(timag.shape)
print(timag[0])

# %%
import torch
batch_size = 10
n_classes = 5
h, w = 24, 24
labels = torch.empty(batch_size, 1, h, w, dtype=torch.long).random_(n_classes)
print(torch.max(labels))
labels = labels[labels!=4]
print(torch.max(labels))
# %%
import cv2
import matplotlib.pyplot as plt
import os
import torch
import os
from os import listdir

ima = "/share/ogarces/PRANC/datasets/iSAID/train/Semantic_masks/images1/P2759_07799_instance_color_RGB.png"
imar = cv2.imread(ima)
print(imar[0,0,:])
#print(imar[0].shape)
imar[0,0,:] = [1,1,1]
print(imar[0,0,:])
print(imar.shape)


# %%
