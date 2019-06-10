from torchvision import transforms
from PIL import Image
import numpy as np
import torch

def process_image(image_path):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    max_size = 256
    target_size = 224
    decimal_place = 0
    
    image = Image.open(image_path)
    
    h = image.height
    w = image.width
    if w > h: 
        factor = w / h
        image = image.resize(size = (int(round(factor*max_size, decimal_place)), max_size))
    else:
        factor = h / w
        image = image.resize(size=(max_size, int(round(factor*max_size, decimal_place))))

    h = image.height
    w = image.width
    image = image.crop(((w - target_size) // 2,
                        (h - target_size) // 2,
                        (w + target_size) // 2,
                        (h + target_size) // 2))

    np_image = np.array(image) / 255

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_image = (np_image - mean) / std 
    np_image = np.transpose(np_image, (2, 0, 1))

    tensor_image = torch.from_numpy(np_image).type(torch.FloatTensor)
    return(tensor_image)