import argparse
import torch
import torch.nn.functional as F
import numpy as np
from create_model import create_model
from get_device import get_device
from process_image import process_image
from imshow import imshow
from load_json import load_json

def main():
    predict_args = get_input_args()
    device = get_device(predict_args.gpu);
    model = load_model(predict_args.model_path, device)
    show_prediction(predict_args.image_path, model, predict_args.top_k, predict_args.category_names, device)

def get_input_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('image_path', type=str, help='The path to the image for inference')
    parser.add_argument('model_path', type=str, help='The path to the pretrained model')
    parser.add_argument('--top_k', type=int, default='5', help='Number of top classes to be predicted')
    parser.add_argument('--category_names', type=str, default='cat_to_name.json', help='JSON file describing the category names')
    parser.add_argument('--gpu', nargs='?', const='gpu', type=str, help='Use GPU (instead of CPU)')
    
    return parser.parse_args()

def load_model(model_path, device):
    checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
    structure = checkpoint['structure']
    hidden_layer1 = checkpoint['hidden_layer1']
    model, _, _ = create_model(device, structure, 0.2, hidden_layer1)
    model.class_to_idx = checkpoint['class_to_idx']
    model.load_state_dict(checkpoint['state_dict'])
    return model

def predict(image_path, model, topk, device):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    img = process_image(image_path)
    img_torch = img.unsqueeze_(0)
    img_torch = img_torch.float()
    
    with torch.no_grad():
        output = model.forward(img_torch.to(device))
    
    probability = F.softmax(output, dim=1)
    
    probs, indices = probability.topk(topk)
    probs = probs.to(device).numpy()[0]
    indices  = indices.to(device).numpy()[0]

    idx_to_class = {v:k for k, v in model.class_to_idx.items()}
    classes = [idx_to_class[x] for x in indices]
    
    # Print results as required
    print(f"Probabilities: {probs}")
    print(f"Classes: {classes}")

    return probs, classes

def show_prediction(img_path, model, topk, json, device):
    cat_to_name = load_json(json)
    #print(cat_to_name)
    #print(cat_to_name["42"])
    probabilities, indices_of_flowers = predict(img_path, model, topk, device)
    #indices_of_flowers = np.array(probabilities[1][0])
    names_of_flowers = [cat_to_name[str(index)] for index in indices_of_flowers]
    percentage = float(probabilities[0]*100)
    percentage = format(percentage, '.2f')
    image = process_image(img_path)
    name_of_flower = cat_to_name[str(indices_of_flowers[0])]
    print(f"With a probability of {percentage}% this image shows a {name_of_flower}.")

if __name__ == "__main__":
    main()