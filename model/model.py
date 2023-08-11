# coding=utf8
import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self):
        super().__init__()

        # lstm layers
        self.lstm = nn.LSTM(input_size=119, hidden_size=128, num_layers=1, batch_first=True)
        self.fc1 = nn.Linear(128, 1)

    def forward(self, x):
        # h0: (number of hidden layers, batch size, hidden size)
        h0 = torch.zeros(1, x.size(0), 128).cuda()
        c0 = torch.zeros(1, x.size(0), 128).cuda()
        x, _ = self.lstm(x, (h0, c0))
        x = x[:, -1, :]
        x = self.fc1(x)
        return x

