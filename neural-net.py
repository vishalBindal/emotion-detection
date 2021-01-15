#!/usr/bin/env python
# coding: utf-8


import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import TensorDataset, DataLoader, random_split
from sklearn.metrics import f1_score
import copy

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(device)

# torch.manual_seed(42)
# np.random.seed(42)


# Import data
train_data = np.genfromtxt('./datasets/train.csv', delimiter=',')
y_train = train_data[:,0]
x_train = train_data[:,1:]
print(x_train.shape)

test_data = np.genfromtxt('./datasets/public_test.csv', delimiter=',')
y_test = test_data[:,0]
x_test = test_data[:,1:]
print(x_test.shape)


# Visualising data
#i=16
#plt.imshow(x_train[i].reshape((48,48)), cmap='magma')
#print(y_train[i])


# decomment this to get gabor filter
"""
from skimage.filters import gabor

def apply_gabor(image):
    temp_image = image.reshape((48,48))
    filt_real, filt_imag = gabor(temp_image, frequency=0.6)
    filt_real = filt_real.reshape(2304)
    return filt_real

def get_gabor_features(data):
    return np.array([apply_gabor(xi) for xi in data])

x_train = get_gabor_features(x_train)
x_test = get_gabor_features(x_test)
"""

# decomment this to get histogram of oriented gradients filter
"""
from skimage.feature import hog

def apply_hog(image):
    temp_image = image.reshape((48,48))
    return hog(temp_image)

def get_hog_features(data):
    return np.array([apply_hog(xi) for xi in data])

x_train = get_hog_features(x_train)
x_test = get_hog_features(x_test)
"""

x_train = torch.tensor(x_train, dtype=torch.float).to(device)
y_train = torch.tensor(y_train, dtype=torch.long).to(device)
x_test = torch.tensor(x_test, dtype=torch.float).to(device)
y_test = torch.tensor(y_test, dtype=torch.long).to(device)



def accuracy(yhat, y):
    preds = torch.argmax(yhat, dim=1)
    return 100*f1_score(y.to(torch.device('cpu')), preds.to(torch.device('cpu')), average='macro')

class Vanilla_nn(nn.Module):
    """
    Vanilla neural network with one hidden layer of 100 perceptrons
    Hidden layer has ReLu activation
    """
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(2304, 100)
        self.output = nn.Linear(100, 7)
        
    def forward(self, xb):
        xb = self.hidden(xb)
        xb = F.relu(xb)
        return self.output(xb)

    
def fit(model, x_train, y_train, learning_rate, epochs, batch_size, epsilon):
    """
    Fitting the dataset to learn parameters of the model
    The loss on validation set is printed after each epoch to detect overfitting
    SGD is used for gradient descent
    """
    # Divide train set into train and val set
    # m = x_train.shape[0]
    # val_size = int(0.3*m) # 3:7 split on validation:train
    # train_ds = TensorDataset(x_train, y_train)
    # val_subset, train_subset = random_split(train_ds, [val_size, m - val_size])
    # train_dl = DataLoader(train_subset, batch_size=batch_size, shuffle=True) # shuffle train dataset
    # val_dl = DataLoader(val_subset, batch_size=2*batch_size) # set greater batch size since backprop not needed

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

        # with torch.no_grad():
            # val_loss = sum(loss_func(model(xb), yb) for xb, yb in val_dl)/len(val_dl)
            # print(epoch, val_loss)
            # if val_loss < cur_loss:
                # cur_loss = val_loss
                # print(f'Final model updated at {epoch} epochs')
                # final_model = copy.deepcopy(model)
    
    # final_model.to(device)
    # return final_model
    return model


for lr in [0.01, 0.05, 0.1]:
    for batch_size in [50, 100, 500]:
        print('Learning rate:', lr)
        print('Batch size:', batch_size)
        print('-------------------')
        model = Vanilla_nn().to(device)

        # Fit data on model
        model = fit(model, x_train, y_train, learning_rate=lr, epochs=100, batch_size=batch_size, epsilon=1e-4)

        # print('Final model (for minimum val loss)')
        # f1 = accuracy(final_model(x_train), y_train)
        # print('Train f-1:', f1)
        # f1 = accuracy(final_model(x_test), y_test)
        # print('Test f-1:', f1)
        # print('Overfitted model (for full epochs)')
        f1 = accuracy(model(x_train), y_train)
        print('Train f-1:', f1)
        f1 = accuracy(model(x_test), y_test)
        print('Test f-1:', f1)
		
		
class Conv_nn(nn.Module):
    """
    A Convolutional neural network with layers as specified in 1c

    """

    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv_layers = nn.Sequential(nn.Conv2d(1, 64, kernel_size=(3,3), stride=3, padding=0),
                                    nn.BatchNorm2d(64),
                                    nn.ReLU(inplace=True),
                                    nn.MaxPool2d(kernel_size=(2,2), stride=2, padding=0),
                                    nn.Conv2d(64, 128, kernel_size=(2,2), stride=2, padding=0),
                                    nn.BatchNorm2d(128),
                                    nn.ReLU(inplace=True),
                                    nn.MaxPool2d(kernel_size=(2,2), stride=2, padding=0),
                                    )
        self.drop_out = nn.Dropout()
        self.fc1 = nn.Linear(12 * 12 * 64, 256)
        self.fc2 = nn.Linear(256, 7)

    def forward(self, x):
        out = self.conv_layers(x)
        out = out.reshape(out.size(0), -1)
        out = self.drop_out(out)
        out = self.fc1(out)
        out = self.fc2(out)
        return out





