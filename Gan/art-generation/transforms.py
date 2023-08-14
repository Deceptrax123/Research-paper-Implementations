import torch 
import torchvision 
import torchvision.transforms as T 
import matplotlib.pyplot as plt 
import numpy as np 

from PIL import Image 


img=Image.open("./Data/Abstract_gallery/Abstract_gallery/Abstract_image_431.jpg")


tensor=T.ToTensor()
i=tensor(img)

print(i.shape)

mean,std=i.mean([1,2],keepdim=True),i.std([1,2],keepdim=True)

composed=T.Compose([T.Resize(size=(128,128)),T.ToTensor(),T.Normalize(mean=mean,std=std)])
img_norm=composed(img)

img_norm=np.array(img_norm)
img_norm=img_norm.transpose(1,2,0)

plt.imshow(img_norm)
plt.show()