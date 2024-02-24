
import torch
import torch.nn.functional as F
from torch import nn
from torch import optim
from torchvision import datasets, transforms, models
import numpy as np
import os

def classifier(mod, dirs, rate, device):
    data_dir = dirs
    train_dir = os.path.join(data_dir , 'valid')
    valid_dir = os.path.join(data_dir , 'valid')
    
    # TODO: Define your transforms for the training, validation, and testing sets
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                      transforms.RandomResizedCrop(224),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])

    test_transforms = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])

    # TODO: Load the datasets with ImageFolder
    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    valid_data = datasets.ImageFolder(valid_dir,transform=test_transforms)
 

    # TODO: Using the image datasets and the trainforms, define the dataloaders
    # dataloaders = 
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    validloader = torch.utils.data.DataLoader(valid_data, batch_size=64, shuffle=True)
   
    model = models.__dict__[mod](pretrained=True)

    if device == 'cuda' and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    for param in model.parameters():
        param.requires_grad = False
    
    
    from collections import OrderedDict
    classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(25088, 256)),
                          ('relu', nn.ReLU()),
                          ('fc2', nn.Linear(256, 128)),
                          ('relu', nn.ReLU()),
                          ('fc3', nn.Linear(128, 102)),
                          ('output', nn.LogSoftmax(dim=1))
    ]))
    criterion = nn.NLLLoss()
    model.classifier = classifier

    optimizer = optim.Adam(model.classifier.parameters(), lr=rate)
    
    return model, device, trainloader, validloader, optimizer, criterion, train_data

    # for epoch in range(epochs):
    # running_loss = 0
    # for inputs, labels in trainloader:
    #     inputs, labels = inputs.to(device), labels.to(device)
    #     output = model.forward(inputs)
    #     loss = criterion(output, labels)
        
    #     optimizer.zero_grad()
    #     loss.backward()
    #     optimizer.step()
        
    #     running_loss += loss.item()

    # else:
    #     valid_loss = 0
    #     accuracy = 0
    #     with torch.no_grad():
    #         model.eval()
    #         for inputs, labels in validloader:
    #             inputs, labels = inputs.to(device), labels.to(device)
    #             output = model.forward(inputs)
    #             valid_loss += criterion(output,labels)
                
    #             ps = torch.exp(output)
    #             top, top_class = ps.topk(1, dim=1)
    #             equals = top_class == labels.view(*top_class.shape)
    #             accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
        
    #     print("Epoch: {}/{}.. ".format(epoch+1, epochs),
    #           "Training Loss: {:.3f}.. ".format(running_loss/len(trainloader)),
    #           "Valid Loss: {:.3f}.. ".format(valid_loss/len(validloader)),
    #           "Test Accuracy: {:.3f}".format(accuracy/len(validloader)))
    #     model.train()
 
