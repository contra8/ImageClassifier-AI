import torch

def get_device(gpu):
    device = torch.device("cuda" if torch.cuda.is_available() and gpu=='gpu' else "cpu")
    return device