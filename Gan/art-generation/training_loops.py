import torch
from torch.utils.data import DataLoader
from data_generator import Dataset
from discriminator import Discriminator
from generator import Generator
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from torch import nn


def train():
    gen_loss=0
    dis_loss=0
    for step,data in enumerate(train_loader):
        #generate the labels
        real_samples=data.to(device=device)
        real_labels=torch.ones((params['batch_size'],1)).to(device=device)

        #generate latent space noise
        latent_space_samples=torch.randn((params['batch_size'],100)).to(device=device)
        generated_samples=generator(latent_space_samples)

        generated_labels=torch.zeros((params['batch_size'],1)).to(device=device)

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
        latent_space_samples=torch.randn((params['batch_size'],100)).to(device=device)
        generator.zero_grad()

        generated_samples=generator(latent_space_samples)
        output_discriminator_generated=discriminator(generated_samples)
        generator_loss=loss_function(output_discriminator_generated,real_labels)
        generator_loss.backward()
        generator_optimizer.step()

        gen_loss+=generator_loss.item()
        dis_loss+=discriminator_loss.item()

        if step==params['batch_size']-1:
            gen_loss=gen_loss/(step+1)
            dis_loss=dis_loss/(step+1)

    return gen_loss,dis_loss

def test():
    gen_loss=0
    dis_loss=0

    for step,data in enumerate(test_loader):
        real_samples=data.to(device=device)
        real_labels=torch.ones((params['batch_size'],1)).to(device=device)

        #generate samples in latent space
        latent_space_samples=torch.randn((params['batch_size'],1)).to(device=device)
        generated_samples=generator(latent_space_samples)

        generated_labels=torch.zeros((params['batch_size'],1)).to(device=device)

        #consider all samples for discriminator loss
        samples=torch.cat((real_samples,generated_samples))
        labels=torch.cat((real_labels,generated_labels))

        #get discriminator loss
        discriminator_output=discriminator(samples)
        loss_discriminator=loss_function(discriminator_output,labels)

        #get generator loss
        latent_space_samples=torch.randn((params['batch_size'],1)).to(device=device)
        generated_samples=generator(latent_space_samples)

        discriminator_generated=discriminator(generated_samples)
        loss_generator=loss_function(discriminator_generated,real_labels)

        gen_loss+=loss_generator.item()
        dis_loss+=loss_discriminator.item()

        if step==params['batch_size']-1:
            gen_loss=gen_loss/(step+1)
            dis_loss=dis_loss/(step+1)
    return gen_loss,dis_loss

def training_loop():

    for epoch in range(num_epochs):
        generator.train(True)
        discriminator.train(True)

        gen_train_loss,dis_train_loss=train()

        generator.eval()
        discriminator.eval()

        gen_test_loss,dis_test_loss=test()
        
        print("Epoch { }".format(epoch+1))
        print("Generator Loss: Train - {} Test- {}".format(gen_train_loss,gen_test_loss))
        print("Discriminator Loss: Train - {} Test-{}".format(dis_train_loss,dis_test_loss))


#initial setup
ids = list(range(0, 2782))
train,test=train_test_split(ids,test_size=0.25)

params={
    'batch_size':8,
    'shuffle':True,
    'num_workers':1
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

#train
training_loop()