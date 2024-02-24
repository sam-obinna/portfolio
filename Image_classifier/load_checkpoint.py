
import torch

from torchvision import datasets, transforms, models
import numpy as np

def load_checkpoint(filepath, mod):
    checkpoint = torch.load(filepath)
    
    # Load the pre-trained model architecture
    model = models.__dict__[mod](pretrained=True)


    # Freeze the pre-trained model parameters
    for param in model.parameters():
        param.requires_grad = False
    
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
    
    # Load the class to index mapping
    model.class_to_idx = checkpoint['class_to_idx']

    return model