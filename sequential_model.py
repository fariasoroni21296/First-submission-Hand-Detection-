import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
from time import time
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch import nn, optim

import torch.nn as nn
import  torch.nn.functional as F

device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.device(device)
seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

def get_transforms():
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),]
    )

    return transform

def get_dataset(download=True):
    transforms = get_transforms()
    trainset = datasets.MNIST('./training_dataset', download = download, train = True, transform = transforms)
    valset = datasets.MNIST('./validation_dataset', download = download, train = False, transform = transforms)

    train_loader = DataLoader(trainset, batch_size=64, shuffle=True)
    val_loader = DataLoader(valset, batch_size=64, shuffle=True)

    return train_loader, val_loader

def plot_data(images):
    figure = plt.figure()
    num_of_images = 60
    for index in range(1, num_of_images + 1):
        plt.subplot(6, 10, index)
        plt.axis('off')
        plt.imshow(images[index].numpy().squeeze(), cmap='gray_r')

def train(model, optimizer, train_loader, train_losses, train_counter, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        data = torch.tensor(data).to(device)
        target = torch.tensor(target).to(device)
        output = model(data)
        loss = nn.NLLLoss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, batch_idx * len(data), len(train_loader.dataset), 
            100. * batch_idx / len(train_loader), 
            loss.item()))
            train_losses.append(loss.item())
            train_counter.append((batch_idx*64) + ((epoch-1)*len(train_loader.dataset)))

        torch.save(model.state_dict(), './results/sequential_digit_recognizer.pth')
        torch.save(optimizer.state_dict(), './optimizer/sequential_optimizer.pth')

        return train_losses, train_counter, True

def test(model, test_loader, test_losses, test_counter):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data = torch.tensor(data).to(device)
            target = torch.tensor(target).to(device)
            output = model(data)
            test_loss += nn.NLLLoss(output, target, size_average=False).item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()
    test_loss /= len(test_loader.dataset)
    test_losses.append(test_loss)

    print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    test_loss, correct, len(test_loader.dataset),
    100. * correct / len(test_loader.dataset)))

    return test_losses, True

if __name__ == '__main__':
    train_loader, val_loader = get_dataset(download = False)
    dataiter = iter(train_loader)

    images, labels = dataiter.next()

    print(images.shape)
    print(labels.shape)
    
    plot_data(images)


    n_epochs = 100
    learning_rate = 0.01
    momentum = 0.9
    batch_size_train = 64
    batch_size_test = 64
    log_interval = 10

    input_size = 784
    hidden_sizes = [128, 64]
    output_size = 10

    model = nn.Sequential(nn.Linear(input_size, hidden_sizes[0]),
                            nn.ReLU(),
                            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
                            nn.ReLU(),
                            nn.Linear(hidden_sizes[1], output_size),
                            nn.LogSoftmax(dim=1))

    #importing model
    if device == 'cuda':
        model.cuda()
        print("GOT CUDA!")
    optimizer = optim.SGD(model.parameters(), lr = learning_rate, momentum = momentum)

    train_losses = []
    test_losses = []

    train_counter = []
    test_counter = [i*len(train_loader.dataset) for i in range(n_epochs + 1)]

    print(model)

    # model_state = torch.load('digit_recognizer.pth', map_location=device)
    # model.load_state_dict(model_state)

    # optimizer_state = torch.load('optimizer.pth', map_location=device)
    # optimizer.load_state_dict(optimizer_state)

    test_losses, is_tested = test(model, val_loader, test_losses, test_counter)
    for epoch in range(n_epochs+1):
        train_loss, train_counter, is_trained = train(model, optimizer, train_loader, train_losses, train_counter, epoch)
        test_losses, is_tested = test(model, val_loader, test_losses, test_counter)
