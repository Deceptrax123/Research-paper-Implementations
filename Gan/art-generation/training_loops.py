import torch
from torch.utils.data import DataLoader
from data_generator import Dataset
from discriminator import Discriminator
from generator import Generator
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from torch import nn
from time import time
import multiprocessing as mp


def train_step():
    gen_loss=0
    dis_loss=0
    for step,data in enumerate(train_loader):
        #generate the labels
        real_samples=data.to(device=device)
        real_labels=torch.ones((real_samples.size(0),1)).to(device=device)

        #generate latent space noise
        latent_space_samples=torch.randn((real_samples.size(0),100)).to(device=device)
        generated_samples=generator(latent_space_samples)

        generated_labels=torch.zeros((real_samples.size(0),1)).to(device=device)

        #consider all samples
        samples=torch.cat((real_samples,generated_samples))
        labels=torch.cat((real_labels,generated_labels))

        #Train the discriminator
        discriminator.zero_grad()
        discriminator_predictions=discriminator(samples)

        discriminator_loss=loss_function(discriminator_predictions,labels)
        discriminator_loss.backward()
        discriminator_optimizer.step()

        #Training the generator
        latent_space_samples=torch.randn((real_samples.size(0),100)).to(device=device)
        generator.zero_grad()

        generated_samples=generator(latent_space_samples)
        output_discriminator_generated=discriminator(generated_samples)
        generator_loss=loss_function(output_discriminator_generated,real_labels)
        generator_loss.backward()
        generator_optimizer.step()

        gen_loss+=generator_loss.item()
        dis_loss+=discriminator_loss.item()

        if step==train_steps-1:
            gen_loss=gen_loss/(step+1)
            dis_loss=dis_loss/(step+1)

    return gen_loss,dis_loss

def test_step():
    gen_loss=0
    dis_loss=0

    for step,data in enumerate(test_loader):
        real_samples=data.to(device=device)
        real_labels=torch.ones((real_samples.size(0),1)).to(device=device)

        #generate samples in latent space
        latent_space_samples=torch.randn((real_samples.size(0),1)).to(device=device)
        generated_samples=generator(latent_space_samples)

        generated_labels=torch.zeros((real_samples.size(0),1)).to(device=device)

        #consider all samples for discriminator loss
        samples=torch.cat((real_samples,generated_samples))
        labels=torch.cat((real_labels,generated_labels))

        #get discriminator loss
        discriminator_output=discriminator(samples)
        loss_discriminator=loss_function(discriminator_output,labels)

        #get generator loss
        latent_space_samples=torch.randn((real_samples.size(0),1)).to(device=device)
        generated_samples=generator(latent_space_samples)

        discriminator_generated=discriminator(generated_samples)
        loss_generator=loss_function(discriminator_generated,real_labels)

        gen_loss+=loss_generator.item()
        dis_loss+=loss_discriminator.item()

        if (step==test_steps-1):
            gen_loss=gen_loss/(step+1)
            dis_loss=dis_loss/(step+1)
    return gen_loss,dis_loss

def training_loop():

    for epoch in range(num_epochs):
        generator.train(True)
        discriminator.train(True)

        train_losses=train_step()

        generator.eval()
        discriminator.eval()

        test_losses=test_step()
        
        print("Epoch { }".format(epoch+1))
        print("Generator Loss: Train - {} Test- {}".format(train_losses[0],train_losses[1]))
        print("Discriminator Loss: Train - {} Test-{}".format(test_losses[0],test_losses[1]))

if __name__=='__main__':
    #initial setup
    ids = list(range(0, 2782))
    train,test=train_test_split(ids,test_size=0.25)

    params={
        'batch_size':8,
        'shuffle':True,
        'num_workers':4
    }

    train_dataset=Dataset(train)
    test_dataset=Dataset(test)

    train_loader=DataLoader(train_dataset,**params)
    test_loader=DataLoader(test_dataset,**params)

    #device usage 
    if torch.backends.mps.is_available():
        device=torch.device("mps")
    else:
        device=torch.device("cpu")

    #get the models
    generator=Generator().to(device=device)
    discriminator=Discriminator().to(device=device)

    #hyperparameters
    lr=0.0002
    num_epochs=100
    loss_function=nn.BCELoss()

    #set optimizer
    generator_optimizer=torch.optim.Adam(generator.parameters(),lr=lr,betas=(0.5,0.999))
    discriminator_optimizer=torch.optim.Adam(discriminator.parameters(),lr=lr,betas=(0.5,0.999))


    train_steps=(len(train)+params['batch_size']-1)//params['batch_size']
    test_steps=(len(test)+params['batch_size']-1)//params['batch_size']

    training_loop()

    # for num_workers in range(2, mp.cpu_count()+2, 2):  
    #     tloader = DataLoader(train_dataset,shuffle=True,num_workers=num_workers,batch_size=32,pin_memory=True)
    #     start = time()
    #     for epoch in range(1, 3):
    #         for i, data in enumerate(tloader,0):
    #             pass
    #     end = time()
    #     print("Finish with:{} second, num_workers={}".format(end - start, num_workers))