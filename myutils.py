import torch
from torch import optim, nn
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from PIL import Image
import model_builder

def load_data(data_dir='flowers'):
    
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    train_transform = transforms.Compose([transforms.RandomRotation(30),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])

    test_transform = transforms.Compose([transforms.Resize(255),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])

    train_data = datasets.ImageFolder(train_dir, transform=train_transform)
    valid_data = datasets.ImageFolder(valid_dir, transform=test_transform)
    test_data = datasets.ImageFolder(test_dir, transform=test_transform)

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=64)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=64)
    
    return train_loader, valid_loader, test_loader, train_data

def save_checkpoint(model, optimiser, path='checkpoint.pth', arch='vgg13', hidden_units=1024, epochs=5):
    checkpoint = {'arch': arch,
                  'hidden_units': hidden_units,
                  'state_dict': model.state_dict(),
                  'class_to_idx': model.class_to_idx,
                  'optimizer_state': optimiser.state_dict(),
                  'epochs': epochs}

    torch.save(checkpoint, path)
    
def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model, optimiser = model_builder.create_model(arch=checkpoint['arch'], hidden_units=checkpoint['hidden_units'])
    model.class_to_idx = checkpoint['class_to_idx']
    model.load_state_dict(checkpoint['state_dict'])
    optimiser.load_state_dict(checkpoint['optimizer_state'])
    return model, optimiser

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    pil_image = Image.open(image)
    
    preprocess = transforms.Compose([transforms.Resize(256),
                                     transforms.CenterCrop(224),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], 
                                                          [0.229, 0.224, 0.225])])
    image_tensor = preprocess(pil_image)
    return image_tensor