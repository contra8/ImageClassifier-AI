import sys
import argparse
import numpy as np
import torch
from torch import nn, optim
import torchvision
from torchvision import transforms, datasets, models
from collections import OrderedDict
from create_model import create_model
from get_device import get_device
from load_json import load_json

def main():
    try:
        data_sets = get_data_sets(sys.argv[1])
    except:
        print("Please give the path/name of an existing directory with your training images (e.g. 'python train.py flowers').")
        exit()
        
    train_args = get_input_args()
    cat_to_name, output_layer_size = load_json('cat_to_name.json')
    device = get_device(train_args.gpu);
    model, optimizer, criterion = create_model(device, train_args.arch, 0.2, train_args.hidden_units, train_args.learning_rate, output_layer_size)
    model = train_the_model(model, train_args.epochs, data_sets[0], data_sets[1], device, optimizer, criterion)
    test_the_model(model, data_sets[2], device)
    save_checkpoint(model, train_args.arch, train_args.hidden_units, data_sets[3], train_args.save_dir)

def get_input_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('images_directory', type=str, help='The path to images for training')
    parser.add_argument('--save_dir', type=str, default='prediction/checkpoint.pth', help='Path to the directory and file to save the checkpoint. Default: prediction/checkpoint.pth')
    parser.add_argument('--arch', type=str, default='vgg16', help='Type of deep learning model. Supported are vgg16 (default) and densenet121.')
    parser.add_argument('--learning_rate', type=float, default='0.001', help='The learning rate')
    parser.add_argument('--hidden_units', type=int, default='5000', help='Number of units in hidden layer')
    parser.add_argument('--epochs', type=int, default='3', help='Name of epochs')
    parser.add_argument('--gpu', nargs='?', const='gpu', type=str, help='Use GPU (instead of CPU)')
    
    return parser.parse_args()

def get_data_sets(data_dir):
    train_dir = data_dir + '/train/'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.RandomResizedCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406],
                                                        [0.229, 0.224, 0.225])])

    validate_transforms = transforms.Compose([transforms.Resize(255),
                                                transforms.CenterCrop(224),
                                                transforms.ToTensor(),
                                                transforms.Normalize([0.485, 0.456, 0.406],
                                                                    [0.229, 0.224, 0.225])])

    test_transforms = transforms.Compose([transforms.Resize(255),
                                            transforms.CenterCrop(224),
                                            transforms.ToTensor(),
                                            transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])

    train_dataset = datasets.ImageFolder(train_dir, transform = train_transforms)
    validate_dataset = datasets.ImageFolder(valid_dir, transform = validate_transforms)
    test_dataset = datasets.ImageFolder(test_dir, transform = test_transforms)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = 64, shuffle = True)
    validate_loader = torch.utils.data.DataLoader(validate_dataset, batch_size = 32, shuffle = True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = 32, shuffle = True)
    
    return train_loader, validate_loader, test_loader, train_dataset

def train_the_model(model, number_epochs, train_loader, validate_loader, device, optimizer, criterion):
    epochs = number_epochs
    train_loader = train_loader
    print_every = 5
    steps = 0

    for epoch in range(epochs):
        running_loss = 0
        for ii, (inputs, labels) in enumerate(train_loader):
            steps += 1
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            if steps % print_every == 0:
                accuracy = 0
                validation_loss = 0
                model.eval()

                with torch.no_grad():
                    for ii, (inputs_val, labels_val) in enumerate(validate_loader):
                        optimizer.zero_grad()
                        inputs_val, labels_val = inputs_val.to(device), labels_val.to(device)
                        model.to(device)
                        outputs_val = model.forward(inputs_val)
                        validation_loss = criterion(outputs_val, labels_val)

                        # Calculate accuracy
                        ps = torch.exp(outputs_val)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels_val.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                print(f"Epoch {epoch+1}/{epochs}.. "
                      f"Train loss: {running_loss/print_every:.3f}.. "
                      f"Validation loss: {validation_loss/len(validate_loader):.3f}.. "
                      f"Validation accuracy: {accuracy/len(validate_loader):.3f}")
                running_loss = 0
                model.train()

        else:
            print(f"Finished epoch {epoch+1}/{epochs} --------")
    return model
            
def test_the_model(model, test_loader, device):
    num_correct = 0
    num_total = 0
    with torch.no_grad():
        for images_test, labels_test in test_loader:
            images_test, labels_test = images_test.to(device), labels_test.to(device)
            outputs = model(images_test)
            _, predicted = torch.max(outputs.data, 1)
            num_total += labels_test.size(0)
            num_correct += (predicted == labels_test).sum().item()

    print('Accuracy: %d %%'% (100 * num_correct / num_total))

def save_checkpoint(model, arch, hidden_units, train_dataset, save_path):
    model.class_to_idx = train_dataset.class_to_idx
    torch.save({'structure': arch,
                'hidden_layer1': hidden_units,
                'state_dict': model.state_dict(),
                'class_to_idx': model.class_to_idx},
                save_path)

if __name__ == "__main__":
    main()