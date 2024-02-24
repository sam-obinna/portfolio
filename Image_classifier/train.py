# Imports here
# %matplotlib inline
# %config InlineBackend.figure_format = 'retina'

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch import nn
from torch import optim
from PIL import Image
from torchvision import datasets, transforms, models
import numpy as np


from get_input import get_input
from model import classifier
from save import save_checkpoint
def main():
    args = get_input()
    
    data_dir = args.dir
    mod = args.arch
    
    rate = args.lr
    epochs = args.epoch
    device = args.gpu
    check_save = args.kdir
   

    model, device, trainloader, validloader, optimizer, criterion, train_data = classifier(mod, data_dir, rate, device)

    model.to(device)
    for epoch in range(epochs):
        running_loss = 0
        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)
            output = model.forward(inputs)
            loss = criterion(output, labels)
        
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
            running_loss += loss.item()

        else:
            valid_loss = 0
            accuracy = 0
            with torch.no_grad():
                model.eval()
                for inputs, labels in validloader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    output = model.forward(inputs)
                    valid_loss += criterion(output,labels)
                
                    ps = torch.exp(output)
                    top, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
        
            print("Epoch: {}/{}.. ".format(epoch+1, epochs),
              "Training Loss: {:.3f}.. ".format(running_loss/len(trainloader)),
              "Valid Loss: {:.3f}.. ".format(valid_loss/len(validloader)),
              "Valid Accuracy: {:.3f}".format(accuracy/len(validloader)))
            model.train()


    save_checkpoint(model, epochs, optimizer, check_save, train_data)
    

if __name__ == "__main__":
    main()

