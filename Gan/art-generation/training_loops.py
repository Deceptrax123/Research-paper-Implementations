import torch
from data_generator import Dataset
import numpy as np
import matplotlib.pyplot as plt

ids = list(range(0, 2782))

params = {
    'batch_size': 1,
    'shuffle': True,
    'num_workers': 0
}

partition = ids

dataset = Dataset(partition)

generator = torch.utils.data.DataLoader(dataset, **params)

for s in generator:
    plt.imshow(s[0])

    plt.show()
