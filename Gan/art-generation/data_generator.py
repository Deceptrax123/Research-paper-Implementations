import torch
from skimage import io


class Dataset(torch.utils.data.Dataset):
    def __init__(self, list_ids):
        self.list_ids = list_ids

    def __len__(self):
        return len(self.list_ids)

    def __getitem__(self, index):
        id = self.list_ids[index]

        sample = io.imread(
            "./Data/Abstract_gallery/Abstract_gallery/Abstract_image_"+str(id)+".jpg")

        return sample
