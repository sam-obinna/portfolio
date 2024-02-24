import os
import torch

def save_checkpoint(model, epochs, optimizer,check_dir, train_data):

    model.class_to_idx = train_data.class_to_idx

    checkpoint = {
        'input_size':25088,
        'ouput_size':102,
        'hidden_layers':[256, 128],
        'state_dict': model.state_dict(),
        'class_to_idx': model.class_to_idx,
        'classifier':model.classifier,
        'epochs': epochs,
        'optimizer_state_dict': optimizer
        }
    file_name = os.path.join(check_dir, 'checkpoint.pth')
    torch.save(checkpoint, file_name)

    return checkpoint