import torch
import torchvision
import torchvision.transforms as transforms
from torch.distributions.uniform import Uniform
import matplotlib.pyplot as plt
import numpy as np
import os
import math
from sklearn.datasets import load_svmlight_file
from torch.utils.data import Dataset

def is_regression(dataest):
    if dataest == 'abalone' or dataest == 'abalone' or dataest == 'cadata' or dataest == 'cpusmall' or dataest == 'space_ga' :
        return True

class svmlight_data(Dataset):
    def __init__(self, data_name, root_dir='./datasets/', transform=None, target_transform=None):
        self.inputs, self.outputs = load_svmlight_file(root_dir + data_name)
        if len(set(self.outputs))  > 2 and self.outputs.min() > 0:
            self.outputs -= self.outputs.min()
        if is_regression(data_name):
            self.outputs = 100*(self.outputs - self.outputs.min()) / (self.outputs.max() - self.outputs.min())
        #self.inputs = self.inputs.toarray()
        #self.outputs = self.outputs.toarray()
        self.transform = transform
        self.target_transform = target_transform
        self.data_name = data_name

    def __len__(self):
        return len(self.outputs)

    def __getitem__(self, idx):
        sample = torch.tensor(self.inputs[idx].A, dtype=torch.float32).view(-1)
        if is_regression(self.data_name):
            label = torch.tensor(self.outputs[idx], dtype=torch.float32)
        else:
            label = torch.tensor(self.outputs[idx], dtype=torch.long)
        if self.transform:
            sample = self.transform(sample)
        if self.target_transform:
            label = self.target_transform(label)
        return sample, label

def data_tf(x):
    x = np.array(x, dtype='float32') / 255
    x = (x - 0.5) / 0.5 # 标准化处理
    x = x.reshape((-1,)) # 拉平
    x = torch.tensor(x)
    return x

def load_data(data_name, batch_size):
    if data_name == 'CIFAR10':
        transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                                download=True, transform=data_tf)
        testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                              download=True, transform=data_tf)
        trainset, validateset = torch.utils.data.random_split(trainset, [45000, 5000])
        validateloader = torch.utils.data.DataLoader(validateset, batch_size=5000, shuffle=True)

        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
        testloader = torch.utils.data.DataLoader(testset, batch_size=10000,
                                                 shuffle=False)
        return trainloader, validateloader, testloader, 3072, 10
    elif data_name == 'mnist':
        trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                                download=True, transform=data_tf)
        testset = torchvision.datasets.MNIST(root='./data', train=False,
                                               download=True, transform=data_tf)
        trainset, validateset = torch.utils.data.random_split(trainset, [54000, 6000])
        validateloader = torch.utils.data.DataLoader(validateset, batch_size=6000, shuffle=True)

        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
        testloader = torch.utils.data.DataLoader(testset, batch_size=10000,
                                                 shuffle=False)
        return trainloader, validateloader, testloader, 784, 10
    else:
        trainset = svmlight_data(data_name)
        trainset, testset = torch.utils.data.random_split(trainset, [int(len(trainset)*0.8), len(trainset) - int(len(trainset)*0.8)])
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
        testloader = torch.utils.data.DataLoader(testset, batch_size=len(testset), shuffle=True)
        feature_size = trainset.dataset.inputs.shape[1]
        class_size = len(set(trainset.dataset.outputs))
        return trainloader, testloader, testloader, feature_size, 1 if is_regression(data_name) else class_size

def empirical_loss(y_pred, y, loss_type):
    if loss_type == 'mse':
        criterion = torch.nn.MSELoss()
        return criterion(y_pred.view(-1), y)
    elif loss_type == 'hinge':
        y_pred = torch.tensor([1 if y > 0 else -1 for y in y_pred], dtype=torch.float)
        print(y_pred.shape, y.shape)
        criterion = torch.nn.MultiMarginLoss()
        return criterion(y_pred, y)
    elif loss_type == "cross_entroy":
        criterion = torch.nn.CrossEntropyLoss()
        return criterion(y_pred, y)
    else:
        print('Undefined loss function!\n')
        os._exit()    

def matplotlib_imshow(img, one_channel=False):
    if one_channel:
        img = img.mean(dim=0)
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    if one_channel:
        plt.imshow(npimg, cmap="Greys")
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))