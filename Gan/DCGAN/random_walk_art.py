#random walk into the latent space from the DCGAN paper

import torch 
import numpy as np 
import matplotlib.pyplot as plt 
import torchvision 
from generator import Generator
import torchvision.transforms as T
from PIL import Image

if __name__=='__main__':

    device=torch.device('mps')

    model=Generator().to(device=device)
    model.load_state_dict(torch.load("./models/abstract_art/generator125.pth"))

    z1=torch.randn((1,100)).to(device=device) #noise
    z2=torch.randn((1,100)).to(device=device) #noise
    z3=torch.randn((1,100)).to(device=device) #noise

    vec1=model(z1)
    vec2=model(z2) 
    vec3=model(z3)

    #vector arithmetic
    v=vec3+vec2-vec1

    v2=vec1+vec2-vec3

    v3=vec1-vec2+vec3


    vec1=vec1.to(device='cpu')
    vec2=vec2.to(device='cpu')
    vec3=vec3.to(device='cpu')
    v=v.to(device='cpu')
    v2=v2.to(device='cpu')
    v3=v3.to(device='cpu')

    v=v.detach().numpy()
    v2=v2.detach().numpy()
    vec1=vec1.detach().numpy()
    vec2=vec2.detach().numpy()
    vec3=vec3.detach().numpy()
    v3=v3.detach().numpy()


    #transpose
    v=v.transpose(0,3,2,1)
    v2=v2.transpose(0,3,2,1)
    vec1=vec1.transpose(0,3,2,1)
    vec2=vec2.transpose(0,3,2,1)
    vec3=vec3.transpose(0,3,2,1)
    v3=v3.transpose(0,3,2,1)

    #plot
    dpi=75
    fig=plt.figure(figsize=(10,10),dpi=dpi)

    ax1=fig.add_subplot(2,3,1)
    ax1.imshow(vec1[0])

    ax2=fig.add_subplot(2,3,2)
    ax2.imshow(vec2[0])

    ax3=fig.add_subplot(2,3,3)
    ax3.imshow(vec3[0])

    ax4=fig.add_subplot(2,3,4)
    ax4.imshow(v[0])

    ax5=fig.add_subplot(2,3,5)
    ax5.imshow(v2[0])

    ax6=fig.add_subplot(2,3,6)
    ax6.imshow(v3[0])

    plt.show()
