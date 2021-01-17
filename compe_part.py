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
import csv
from torchvision.models import resnet50, resnet18, resnet34, resnet101, resnet152, resnext50_32x4d, resnext101_32x8d
from torchvision.transforms import Compose, Resize, Grayscale, ToTensor, Normalize
import copy

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(device)

# torch.manual_seed(42)
# np.random.seed(42)

from skimage.transform import rotate, AffineTransform, warp

def get_rotated(image,angle):
  temp_image = image.reshape((48,48))
  return rotate(temp_image,angle-angle,mode='wrap').reshape(2304)
 
def rotated(data,angle):
  return np.array([get_rotated(xi,angle) for xi in data])


# Import data
# train_data = np.genfromtxt('./datasets/train.csv', delimiter=',')
train_data = np.genfromtxt('./datasets/debug.csv', delimiter=',')
y_train = train_data[:,0]
x_train = train_data[:,1:]
print(x_train.shape)

x_train = np.concatenate((x_train,rotated(x_train,10)) , axis=0)
y_train = np.concatenate((y_train,y_train) , axis=0)

print(x_train.shape)

# test_data = np.genfromtxt('./datasets/public_test.csv', delimiter=',')
# test_data = np.genfromtxt('./datasets/private.csv', delimiter=',')
test_data = np.genfromtxt('./datasets/debug.csv', delimiter=',')
y_test = test_data[:,0]
x_test = test_data[:,1:]
print(x_test.shape)

x_test = np.concatenate((x_test,rotated(x_test,10),rotated(x_test,5)) , axis=0)
y_test = np.concatenate((y_test,y_test,y_test) , axis=0)

print(x_test.shape)

# test_pvt = np.genfromtxt('./datasets/private.csv', delimiter=',')
test_pvt = np.genfromtxt('./datasets/debug.csv', delimiter=',')
x_pvt = test_pvt[:,1:]
print(x_pvt.shape)

x_train = torch.tensor(x_train, dtype=torch.float).to(device)
y_train = torch.tensor(y_train, dtype=torch.long).to(device)
x_test = torch.tensor(x_test, dtype=torch.float).to(device)
y_test = torch.tensor(y_test, dtype=torch.long).to(device)
x_pvt = torch.tensor(x_pvt, dtype=torch.float).to(device)


def accuracy(preds, y):
    return 100*f1_score(y.to(torch.device('cpu')), preds.to(torch.device('cpu')), average='macro')

def predict(model, x_test):
    test_ds = TensorDataset(x_test)
    test_dl = DataLoader(test_ds, batch_size=100)
    preds = None
    for xb in test_dl:
        yhatb = model(xb[0])
        predsb = torch.argmax(yhatb, dim=1)
        if preds is not None:
            preds = torch.cat((preds, predsb), 0)
        else:
            preds = predsb
    return preds
    
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
        # print(epoch, avg_loss)
        if abs(avg_loss - cur_loss) <= epsilon:
            print(epoch, avg_loss)
            break
        cur_loss = avg_loss
        
    return model

def initialize_model(model_name, num_labels=7):
    model = eval(model_name)(pretrained=True)
    # w = torch.zeros((64, 1, 7, 7))
    # nn.init.kaiming_uniform_(w, a=math.sqrt(5))
    
    # model.conv1.weight.data = w
	
    conv_out_features = model.fc.in_features
    model.fc = nn.Linear(conv_out_features, num_labels)
	
    return model
	
def write_to_file(preds, filename):
    preds = preds.to(torch.device('cpu'))
    preds = preds.numpy()
    with open(filename+'-preds.csv', 'w') as preds_file:
        writer = csv.writer(preds_file, delimiter=',')
        writer.writerow(['Id', 'Prediction'])
        for i in range(preds.shape[0]):
            writer.writerow([i+1, int(preds[i])])

# reshape data
x_train = x_train.view(x_train.shape[0], 1, 48, 48)
x_test = x_test.view(x_test.shape[0], 1, 48, 48)
x_pvt = x_pvt.view(x_pvt.shape[0], 1, 48, 48)

# augment flipped data
x_flipped = torch.flip(x_train, [3])
x_train = torch.cat((x_train, x_flipped), 0)
y_train = torch.cat((y_train, y_train), 0)

# Scale all values in [0,1]
max_val = max(torch.max(x_train), torch.max(x_test), torch.max(x_pvt))
x_train = x_train / max_val
x_test = x_test / max_val
x_pvt = x_pvt / max_val

# duplicate image along 3 channels
x_train = torch.cat((x_train, x_train, x_train), 1)
x_test = torch.cat((x_test, x_test, x_test), 1)
x_pvt = torch.cat((x_pvt, x_pvt, x_pvt), 1)

# Normalise for model
normalize = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
x_train = normalize(x_train)
x_test = normalize(x_test)
x_pvt = normalize(x_pvt)

lr = 0.01
batch_size = 100

print('Learning rate:', lr)
print('Batch size:', batch_size)
print('-------------------')

models = ['resnext101_32x8d']
for model_name in models:
    print('Trying model:', model_name)
    model = initialize_model(model_name).to(device)

    # Fit data on model
    model = fit(model, x_train, y_train, learning_rate=lr, epochs=100, batch_size=batch_size, epsilon=1e-4)
    f1 = accuracy(predict(model, x_train), y_train)
    print('Train f-1:', f1)
    f1 = accuracy(predict(model, x_test), y_test)
    print('Test f-1:', f1)
    test_predictions = predict(model, x_pvt)
    write_to_file(test_predictions, model_name)
