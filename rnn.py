import torch
import torch.nn as nn
import torch.nn.functional as F


class RNN(nn.Module):
    def __init__(self, hidden_size, layer_dim=1, num_classes=5, use_lstm=False, input_size=12):
        super(RNN, self).__init__()
        if use_lstm:
            self.rnn = nn.LSTM(input_size, hidden_size, layer_dim, batch_first=True)
        else:
            self.rnn = nn.GRU(input_size, hidden_size, layer_dim, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)


    def forward(self, x):

        _, x = self.rnn(x)
        x = x[0].squeeze(0)
        x = self.fc(x)
        return x
