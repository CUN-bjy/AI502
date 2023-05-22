import matplotlib.pyplot as plt
import numpy as np
import random

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()
        
        self.rnn = nn.RNN(1, 10, batch_first=True)
        self.relu = nn.ReLU()
        self.final = nn.Linear(10, 1)

    def forward(self, inputs):
        outputs, hidden = self.rnn(inputs)
        outputs = outputs[:, -1, :]
        outputs = self.final(outputs)

        return outputs

class MyRNN(nn.Module):
    def __init__(self):
        super(MyRNN, self).__init__()
        self.rnn = nn.RNN(1, 10, batch_first=True)
        self.relu = nn.ReLU()
        self.final = nn.Linear(10, 1)

    def forward(self, inputs):
        outputs, hidden = self.rnn(inputs)
        outputs = outputs[:, -1, :]
        outputs = self.final(outputs)
        return outputs

def train_net(net, data, criterion, epochs, lr_rate):
    optim = optimizer(net.parameters(), lr=lr_rate)
    data_iter = DataLoader(data, batch_size, shuffle=True)

    for epoch in range(1, epochs+1):
        running_loss = 0.0
        
        for x, y in data_iter:
            optim.zero_grad()
            output = net(x)
            loss = criterion(output, y.reshape(-1, 1))
            loss.backward()
            optim.step()
            running_loss += loss.item()
        
        print("epoch: {}, loss: {:.2f}".format(epoch, running_loss))
    
    test_loss = criterion(net(test_data[:][0]), test_data[:][1].reshape(-1,1))
    print('test loss: %f' % test_loss.mean().detach().numpy())

    return net

if __name__=="__main__":
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    T = 1000
    time = torch.arange(0.0, T)
    X = torch.sin(0.01 * time) + 0.2 * torch.randn(T)

    # Hyperparameter
    batch_size, train_number = 16, 600

    # timestamp
    tau = 4
    features = torch.zeros((T-tau, tau, 1)) # RNN Model needs tau inputs: shape is (996, 4, 1)

    for i in range(tau):
        features[:, i] = X[i:(T-tau + i)].reshape(-1, 1)

    labels = X[tau:] # shape is (996, )

    # Prepare DataLoader
    train_data = TensorDataset(features[:train_number, :], labels[:train_number])
    test_data = TensorDataset(features[train_number:, :], labels[train_number:])
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam
    
    my_net = MyRNN()
    net = train_net(my_net, train_data, criterion, 10, 0.01)
    
    estimates = net(features)
    predictions = torch.zeros_like(estimates)
    predictions[:(train_number - tau)] = estimates[:(train_number - tau)]
    
    my_estimates = my_net(features)
    plt.plot(time.numpy(), X.numpy(), label='GT')
    plt.plot(time[tau:].numpy(), my_estimates.detach().numpy(), label='Estimate')
    plt.legend()
    plt.savefig("GTvsEST.png")

    my_train_estimate_loss = criterion(my_estimates[:train_number-tau].squeeze(), labels[:train_number-tau]).item()
    my_test_estimate_loss = criterion(my_estimates[train_number-tau:].squeeze(), labels[train_number-tau:]).item()
    print('train loss:', my_train_estimate_loss)
    print('test loss:', my_test_estimate_loss)
    
    my_predictions = torch.zeros_like(my_estimates)
    my_predictions[:(train_number - tau)] = my_estimates[:(train_number - tau)]

    for i in range((train_number - tau), (T - tau)):
        my_predictions[i] = my_net(
            my_predictions[(i - tau):i].reshape(1, -1, 1)
        )

    plt.cla()
    plt.plot(time.numpy(), X.numpy(), label='GT')
    plt.plot(time[tau:].numpy(), my_predictions.detach().numpy(), label='Prediction')
    plt.legend()
    plt.savefig("GTvsPred.png")

    my_train_prediction_loss = criterion(my_predictions[:train_number-tau].squeeze(), labels[:train_number-tau]).item()
    my_test_prediction_loss = criterion(my_predictions[train_number-tau:].squeeze(), labels[train_number-tau:]).item()
    print('train loss:', my_train_prediction_loss)
    print('test loss:', my_test_prediction_loss)