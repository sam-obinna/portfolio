import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch import nn
from torch import optim
from PIL import Image
from torchvision import datasets, transforms, models
import numpy as np
import json


from get_input import predict_input
from load_checkpoint import load_checkpoint
from process_image import process_image

def main():

    args = predict_input()

    image_path = args.file
    checkpoint = args.checkpoint

    mod = args.arch

    device = args.gpu

    topk = args.topk

    classes = args.clas


    with open(classes, 'r') as f:
        cat_to_name = json.load(f)

    model = load_checkpoint(checkpoint,mod)
    image = process_image(image_path)

    if device == 'cuda' and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    image = torch.from_numpy(np.array([image])).float()
    model = model.to(device)
    image = image.to(device)
    model.eval()
    with torch.no_grad():
        output = model.forward(image)
        
    probability = torch.exp(output)
    
    top_probs, top_indices = probability.topk(topk)
    
    top_probs = torch.tensor(top_probs).view(-1).tolist()

    
    idx_to_class = {v: k for k, v in model.class_to_idx.items()}

    top_classes = [idx_to_class[idx.item()] for idx in top_indices[0]]
    labels = [cat_to_name[class_idx] for class_idx in top_classes]
   
    if len(labels) == len(top_classes) == len(top_probs):
    # Iterate over the elements of the lists
        for i in range(len(labels)):
            print(f"Label: {labels[i]}, Class: {top_classes[i]}, Probability: {top_probs[i]}")
    else:
        print("Error: The lists have different lengths.")

if __name__ == "__main__":
    main()
