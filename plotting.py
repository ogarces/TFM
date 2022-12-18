import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import torch


img = mpimg.imread('/share/ogarces/PRANC/datasets/iSAID/val/Semantic_masks/images/P0003_instance_color_RGB.png')
print(img)
plt.imshow(img)
plt.show()


model = torch.jit.load('/share/ogarces/PRANC/SEGMENTResnet50_iSAID_NORM/best_modelpranc.pt')
model.eval()
print(model('/share/ogarces/PRANC/datasets/iSAID/val/images/images1/P0004_00.png'))
