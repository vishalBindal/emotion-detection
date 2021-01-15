#!/usr/bin/env python
# coding: utf-8

# In[2]:


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


# In[3]:


# Import data
train_data = np.genfromtxt('./datasets/train.csv', delimiter=',')
y_train = train_data[:,0]
x_train = train_data[:,1:]
print(x_train.shape)

test_data = np.genfromtxt('./datasets/public_test.csv', delimiter=',')
y_test = test_data[:,0]
x_test = test_data[:,1:]
print(x_test.shape)


# In[4]:


# Visualising data
#i=16
#plt.imshow(x_train[i].reshape((48,48)), cmap='magma')
#print(y_train[i])


# In[5]:


x_train = torch.tensor(x_train, dtype=torch.float).to(device)
y_train = torch.tensor(y_train, dtype=torch.long).to(device)
x_test = torch.tensor(x_test, dtype=torch.float).to(device)
y_test = torch.tensor(y_test, dtype=torch.long).to(device)


# In[6]:


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

    
def fit(model, x_train, y_train, learning_rate, epochs, batch_size):
    """
    Fitting the dataset to learn parameters of the model
    The loss on validation set is printed after each epoch to detect overfitting
    SGD is used for gradient descent
    """
    # Divide train set into train and val set
    m = x_train.shape[0]
    val_size = int(0.3*m) # 3:7 split on validation:train
    train_ds = TensorDataset(x_train, y_train)
    val_subset, train_subset = random_split(train_ds, [val_size, m - val_size])

    train_dl = DataLoader(train_subset, batch_size=batch_size, shuffle=True) # shuffle train dataset
    val_dl = DataLoader(val_subset, batch_size=2*batch_size) # set greater batch size since backprop not needed

    opt = optim.SGD(model.parameters(), lr=learning_rate)
    loss_func = F.cross_entropy
    
    final_model = model
    cur_loss = float('inf')

    for epoch in range(epochs):
        model.train()
        for xb, yb in train_dl:
            # Forward prop
            loss = loss_func(model(xb), yb)
            # Backward prop
            loss.backward()
            opt.step()
            opt.zero_grad()

        model.eval()
        with torch.no_grad():
            val_loss = sum(loss_func(model(xb), yb) for xb, yb in val_dl)/len(val_dl)
            # print(epoch, val_loss)
            if val_loss < cur_loss:
                cur_loss = val_loss
                print(f'Final model updated at {epoch} epochs')
                final_model = copy.deepcopy(model)
    
    final_model.to(device)
    return final_model

# In[7]:

for lr in [0.01, 0.05, 0.1]:
    for batch_size in [50, 100, 500]:
        print('Learning rate:', lr)
        print('Batch size:', batch_size)
        print('-------------------')
        model = Vanilla_nn().to(device)

        # Fit data on model
        final_model = fit(model, x_train, y_train, learning_rate=0.05, epochs=100, batch_size=100)

        print('Final model (for minimum val loss)')
        f1 = accuracy(final_model(x_train), y_train)
        print('Train f-1:', f1)
        f1 = accuracy(final_model(x_test), y_test)
        print('Test f-1:', f1)
        print('Overfitted model (for full epochs)')
        f1 = accuracy(model(x_train), y_train)
        print('Train f-1:', f1)
        f1 = accuracy(model(x_test), y_test)
        print('Test f-1:', f1)


# In[ ]:




