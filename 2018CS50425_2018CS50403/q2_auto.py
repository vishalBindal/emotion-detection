import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import TensorDataset, DataLoader
import sys
from skimage.feature import hog

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

# hog
def apply_hog(image):
    temp_image = image.reshape((48,48))
    return hog(temp_image)

def get_hog_features(data):
    return np.array([apply_hog(xi) for xi in data])

x_train = get_hog_features(x_train)
x_test = get_hog_features(x_test)

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

class Vanilla_nn(nn.Module):
    """
    Vanilla neural network with one hidden layer of 100 perceptrons
    Hidden layer has ReLu activation
    """
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(1296, 100)
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

model = Vanilla_nn().to(device)

# Fit data on model
model = fit(model, x_train, y_train, learning_rate=lr, epochs=100, batch_size=batch_size, epsilon=1e-4)

preds = predict(model, x_test)
preds = preds.to(torch.device('cpu'))
preds = preds.numpy()
preds = preds.astype(int)
write_predictions(output_path, preds)