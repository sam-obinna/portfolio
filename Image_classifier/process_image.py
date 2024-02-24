import numpy as np

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # TODO: Process a PIL image for use in a PyTorch model
    from PIL import Image
    
    
    im = Image.open(image)
    im.thumbnail((256,256))
    
    width, height = im.size
    left = (width - 224) / 2
    top = (height - 224) / 2
    right = (width + 224) / 2
    bottom = (height + 224) / 2

    im = im.crop((left, top, right, bottom))
    
    np_image = np.array(im)/255
    
    means = np.array([0.485, 0.456, 0.406])
    stds = np.array([0.229, 0.224, 0.225])

    np_image = (np_image - means) / stds
    
#     np_image = np_image.numpy().transpose((2, 0, 1))
    np_image = np.transpose(np_image, (2, 0, 1))
    
    return np_image
    
