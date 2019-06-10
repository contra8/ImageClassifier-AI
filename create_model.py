from torchvision import transforms, datasets, models
from torch import nn, optim
from collections import OrderedDict

def create_model(device, structure = 'vgg16', dropout_rate = 0.2, hidden_layer1 = 5000, lr = 0.001):
    if structure == 'vgg16':
        model = models.vgg16(pretrained=True)
    elif structure == 'densenet':
        model = models.densenet(pretrained=True)
    elif structure == 'alexnet':
        model = models.alexnet(pretrained=True)
    else:
        model = models.vgg16(pretrained=True)
        print("The system does not know the given architecture. Thus it uses pretrained vgg16 as default instead.")
        
    # No backpropagation for now
    for param in model.parameters():
        param.requires_grad = False

    classifier = nn.Sequential(OrderedDict([
        ('fc_1', nn.Linear(25088, hidden_layer1)),
        ('relu_1', nn.ReLU()),
        ('dropout_1', nn.Dropout(dropout_rate)),
        ('fc_2', nn.Linear(hidden_layer1, 102)),
        ('output_layer', nn.LogSoftmax(dim=1))
    ]))
    
    model.classifier = classifier
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr)
    model.to(device)
    
    return model, optimizer, criterion
