import torch
from skimage import io
from PIL import Image
import torchvision 
import torchvision.transforms as T
import numpy as np 

class Dataset(torch.utils.data.Dataset):
    def __init__(self, list_ids):
        self.list_ids = list_ids

    def __len__(self):
        return len(self.list_ids)

    def __getitem__(self, index):
        id = self.list_ids[index]

        sample = Image.open(
            "./Data/Abstract_gallery/Abstract_gallery/Abstract_image_"+str(id)+".jpg")

        #initial transform
        tensor=T.ToTensor()
        sa=tensor(sample)

        mean,std=sa.mean([1,2]),sa.std([1,2])

        #Composed transform
        composed_transforms=T.Compose([T.Resize(size=(64,64)),T.ToTensor(),T.Normalize(mean=mean,std=std)])

        sample=composed_transforms(sample)

        return sample
