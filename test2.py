# import numpy as np
# import os
# from PIL import Image
# imdir = '/share/ogarces/PRANC/mmsegmentation/data/potsdam/ann_dir/val'
# files = os.listdir(imdir)

# maxs = []
# newmaxs = []
# for file in files:
#     f = os.path.join(imdir, file)
#     i = Image.open(f)
#     ma =np.max(np.array(i))
#     mi = np.min(np.array(i))

#     if ma > 6 or mi < 1:
#         maxs.append((ma, mi, file))
        
    


# for m in maxs:

#     print(m)
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import torch

img_batch = np.array(torch.rand([8, 3, 256, 256]))
for i in range(8):
    img_batch[i, 0] = np.arange(0, 65536).reshape(256, 256) / 65536 / 8 * i
    img_batch[i, 1] = (1 - np.arange(0, 65536).reshape(256, 256) / 65536 / 8 * i) / 8 * i

writer = SummaryWriter()
writer.add_images('my_image_batch', img_batch, 0)
writer.close()