import torch 
import torchvision 
import torchvision.transforms as T 
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os 


#experiement python file
sample = Image.open(
            "./Data/Abstract_gallery/Abstract_gallery/Abstract_image_"+str(2766)+".jpg")

#initial transform
tensor=T.ToTensor()
sa=tensor(sample)

mean,std=sa.mean([1,2]),sa.std([1,2])

#Composed transform
composed_transforms=T.Compose([T.Resize(size=(128,128)),T.ToTensor(),T.Normalize(mean=mean,std=std)])

sample=composed_transforms(sample)

sample=np.array(sample)
sample=sample.transpose(1,2,0)

# plt.imshow(sample)
# plt.show()

#get cpu cores count
print(os.cpu_count())