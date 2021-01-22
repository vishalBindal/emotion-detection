import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import TensorDataset, DataLoader
import sys

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

train_path = sys.argv[1]
test_path = sys.argv[2]
output_path = sys.argv[3]

# Import data
train_data = np.genfromtxt(train_path, delimiter=',')
y_train = train_data[:,0]
x_train = train_data[:,1:]

test_data = np.genfromtxt(test_path, delimiter=',')
x_test = test_data[:,1:]

x_train = torch.tensor(x_train, dtype=torch.float).to(device)
y_train = torch.tensor(y_train, dtype=torch.long).to(device)
x_test = torch.tensor(x_test, dtype=torch.float).to(device)

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

class Conv_nn(nn.Module):
    """
    A Convolutional neural network with layers as specified in 1c

    """

    def __init__(self):
        super(Conv_nn, self).__init__()
        self.conv_layers = nn.Sequential(nn.Conv2d(1, 64, kernel_size=3, stride=3, padding=0),
                                    nn.BatchNorm2d(64,momentum=0.99, eps=1e-3),
                                    nn.ReLU(),
                                    nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
                                    nn.Conv2d(64, 128, kernel_size=2, stride=2, padding=0),
                                    nn.BatchNorm2d(128,momentum=0.99, eps=1e-3),
                                    nn.ReLU(),
                                    nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
                                    )

        self.linear_layers = nn.Sequential(nn.Linear(512,256),
                                            nn.BatchNorm1d(256,momentum=0.99, eps=1e-3),
                                            nn.ReLU(),
                                            nn.Linear(256,7),
                                            nn.BatchNorm1d(7,momentum=0.99, eps=1e-3),
                                            nn.ReLU()
                                          )

    def forward(self, x):
        out = self.conv_layers(x)
        # out.shape is (batch_size, 128, 2, 2)
        out = out.view(out.shape[0],512)
        out = self.linear_layers(out)
        return out
    
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
        if abs(avg_loss - cur_loss) <= epsilon:
            break
        cur_loss = avg_loss

    return model

def write_predictions(fname, arr):
    np.savetxt(fname, arr, fmt="%d", delimiter="\n")

lr = 0.1
batch_size = 100

model = Conv_nn().to(device)

# reshape data
x_train = x_train.view(x_train.shape[0], 1, 48, 48)
x_test = x_test.view(x_test.shape[0], 1, 48, 48)

# Fit data on model
model = fit(model, x_train, y_train, learning_rate=lr, epochs=100, batch_size=batch_size, epsilon=1e-4)

preds = predict(model, x_test)
preds = preds.to(torch.device('cpu'))
preds = preds.numpy()
preds = preds.astype(int)
write_predictions(output_path, preds)