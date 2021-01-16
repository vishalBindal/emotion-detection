#!/usr/bin/env python
# coding: utf-8


import numpy as np
import math
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import TensorDataset, DataLoader, random_split
from sklearn.metrics import f1_score

from torchvision.models import resnet50
from torchvision.transforms import Compose, Resize, Grayscale, ToTensor
import copy

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(device)

# torch.manual_seed(42)
# np.random.seed(42)


# Import data
train_data = np.genfromtxt('./datasets/train.csv', delimiter=',')
# train_data = np.genfromtxt('./datasets/debug.csv', delimiter=',')
y_train = train_data[:,0]
x_train = train_data[:,1:]
print(x_train.shape)

test_data = np.genfromtxt('./datasets/public_test.csv', delimiter=',')
# test_data = np.genfromtxt('./datasets/debug.csv', delimiter=',')
y_test = test_data[:,0]
x_test = test_data[:,1:]
print(x_test.shape)


x_train = torch.tensor(x_train, dtype=torch.float).to(device)
y_train = torch.tensor(y_train, dtype=torch.long).to(device)
x_test = torch.tensor(x_test, dtype=torch.float).to(device)
y_test = torch.tensor(y_test, dtype=torch.long).to(device)



def accuracy(yhat, y):
    preds = torch.argmax(yhat, dim=1)
    return 100*f1_score(y.to(torch.device('cpu')), preds.to(torch.device('cpu')), average='macro')

    
def fit(model, x_train, y_train, learning_rate, epochs, batch_size, epsilon):
    """
    Fitting the dataset to learn parameters of the model
    The loss on validation set is printed after each epoch to detect overfitting
    SGD is used for gradient descent
    """

    train_ds = TensorDataset(x_train, y_train)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True) # shuffle train dataset
    
    opt = optim.SGD(model.parameters(), lr=learning_rate)
    loss_func = F.cross_entropy
    
    # final_model = model
    cur_loss = float('inf')

    for epoch in range(epochs):
        model.train()
        avg_loss, count = 0, 0
        for xb, yb in train_dl:
            # Forward prop
            loss = loss_func(model(xb), yb)
            avg_loss, count = avg_loss + loss, count + 1
            # Backward prop
            loss.backward()
            opt.step()
            opt.zero_grad()

        model.eval()

        avg_loss = avg_loss / count
        print(epoch, avg_loss)
        if abs(avg_loss - cur_loss) <= epsilon:
            break
        cur_loss = avg_loss

        
    return model

def initialize_model(num_labels=7):
    model = resnet50(pretrained=True)
    w = torch.zeros((64, 1, 7, 7))
    nn.init.kaiming_uniform_(w, a=math.sqrt(5))
    
    model.conv1.weight.data = w
	
    conv_out_features = model.fc.in_features
    model.fc = nn.Linear(conv_out_features, num_labels)
	
    return model
	

lr = 0.01
batch_size = 100

print('Learning rate:', lr)
print('Batch size:', batch_size)
print('-------------------')
#model = Conv_nn().to(device)
model = initialize_model().to(device)

# reshape data
x_train = x_train.view(x_train.shape[0], 1, 48, 48)
x_test = x_test.view(x_test.shape[0], 1, 48, 48)
# Fit data on model
model = fit(model, x_train, y_train, learning_rate=lr, epochs=100, batch_size=batch_size, epsilon=1e-4)
f1 = accuracy(model(x_train), y_train)
print('Train f-1:', f1)
f1 = accuracy(model(x_test), y_test)
print('Test f-1:', f1)