import torch
from torch.utils.data import DataLoader
from data_generator import AbstractArtDataset
from discriminator import Discriminator
from generator import Generator
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from initialize import initialize_weights
from torch import nn
from time import time
import multiprocessing as mp
import torch.multiprocessing
import wandb

def train_step():
    gen_loss=0
    dis_loss=0
    for step,data in enumerate(train_loader):
        #generate the labels
        real_samples=data.to(device=device)
        real_labels=torch.ones((real_samples.size(0),1)).to(device=device)
        generated_labels=torch.zeros((real_samples.size(0),1)).to(device=device)


        #generate latent space noise
        latent_space_samples=torch.randn((real_samples.size(0),100)).to(device=device)
        generated_samples=generator(latent_space_samples)

        #consider all samples
        samples=torch.cat((real_samples,generated_samples))
        labels=torch.cat((real_labels,generated_labels))

        #Train the discriminator
        discriminator.zero_grad()
        discriminator_predictions=discriminator(samples)
        discriminator_loss=loss_function(discriminator_predictions,labels)
        discriminator_loss.backward()
        discriminator_optimizer.step()

        latent_space_samples=torch.randn((real_samples.size(0),100)).to(device=device)
        generated_samples=generator(latent_space_samples)
        #Training the generator
        generator.zero_grad()
        output_discriminator_generated=discriminator(generated_samples)
        generator_loss=loss_function(output_discriminator_generated,real_labels)
        generator_loss.backward()
        generator_optimizer.step()

        gen_loss+=generator_loss.item()
        dis_loss+=discriminator_loss.item()
    
    gloss=gen_loss/train_steps
    dloss=dis_loss/train_steps

    return gloss,dloss

def training_loop():

    glosses=[]
    dlosses=[]
    for epoch in range(num_epochs):
        generator.train(True)
        discriminator.train(True)

        train_losses=train_step()

        generator.train(False)
        discriminator.train(False) 
        
        print('Epoch {epoch}'.format(epoch=epoch+1))
        print("Generator Loss: {gloss}".format(gloss=train_losses[0]))
        print("Discriminator Loss: {dloss}".format(dloss=train_losses[1]))

        wandb.log({'Generator Loss':train_losses[0],'Discriminator Loss':train_losses[1]})


        dlosses.append(train_losses[1])
        glosses.append(train_losses[0])

        #save model at epoch checkpoints
        if((epoch+1)%25==0):
            path='generator{number}.pth'.format(number=epoch+1)
            torch.save(generator.state_dict(),path)

if __name__=='__main__':
    torch.multiprocessing.set_sharing_strategy('file_system')
    #initial setup
    ids = list(range(0, 2782))

    params={
        'batch_size':128,
        'shuffle':True,
        'num_workers':4
    }

    dataset=AbstractArtDataset(ids)

    wandb.init(
        project="art-generation",
        config={
            "learning_rate":0.0002,
            "architecture":"Adversarial",
            "dataset":"Art generation from kaggle",
            "Epochs":100,
        },
    )

    train_loader=DataLoader(dataset,**params)

    #device usage 
    if torch.backends.mps.is_available():
        device=torch.device("mps")
    else:
        device=torch.device("cpu")
    #get the models
    generator=Generator().to(device=device)
    discriminator=Discriminator().to(device=device)

    initialize_weights(generator)
    initialize_weights(discriminator)

    #hyperparameters
    lr=2e-4
    num_epochs=100
    loss_function=nn.BCEWithLogitsLoss()

    #set optimizer
    generator_optimizer=torch.optim.Adam(generator.parameters(),lr=lr,betas=(0.5,0.999))
    discriminator_optimizer=torch.optim.Adam(discriminator.parameters(),lr=lr,betas=(0.5,0.999))


    train_steps=(len(ids)+params['batch_size']-1)//params['batch_size']

    history=training_loop()

    #get ideal count of num_workers
    # for num_workers in range(2, mp.cpu_count()+2, 2):  
    #     tloader = DataLoader(train_dataset,shuffle=True,num_workers=num_workers,batch_size=32,pin_memory=True)
    #     start = time()
    #     for epoch in range(1, 3):
    #         for i, data in enumerate(tloader,0):
    #             pass
    #     end = time()
    #     print("Finish with:{} second, num_workers={}".format(end - start, num_workers))
