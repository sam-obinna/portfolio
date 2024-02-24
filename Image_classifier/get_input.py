import argparse

def train_input():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dir', type = str, default = 'flowers/', help = 'path to the folder of pet images')

    parser.add_argument('--arch', type = str, default = 'vgg11', help = 'The Pretrained Model')

    parser.add_argument('--gpu', type = str, default = 'cuda.0', help = 'The model loader')

    parser.add_argument('--kdir', type = str, default = './', help = 'The Directory to save the checkpoint')

    parser.add_argument('--epoch', type = int, default = 3, help = 'The number of epochs')

    parser.add_argument('--hidden_units', type = int, default = 512, help = 'The number of epochs')

    parser.add_argument('--lr', type = float, default = 0.003, help = 'Learning Rate')

    args = parser.parse_args()

    

    return args


def predict_input():
    parser = argparse.ArgumentParser()

    parser.add_argument('--file', type = str, default = './flowers/valid/10/image_07101.jpg', help = 'file for training')

    parser.add_argument('--checkpoint', type = str, default = './checkpoint.pth', help = 'The Checkp  Model')

    parser.add_argument('--arch', type = str, default = 'vgg11', help = 'The Pretrained Model')

    parser.add_argument('--gpu', type = str, default = 'cuda', help = 'The model loader')

    parser.add_argument('--topk', type = int, default =5, help = 'The Directory to save the checkpoint')

    parser.add_argument('--clas', type = str, default = 'cat_to_name.json', help = 'The Claasws Mstch')

   
    args = parser.parse_args()

    

    return args