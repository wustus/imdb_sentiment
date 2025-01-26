
from torch import nn


class LSTMNetwork(nn.Module):

    def __init__(self, in_size, hidden_size, out_size):

        super().__init__()

        self.in_size = in_size
        self.lstm = nn.LSTM(in_size, hidden_size, batch_first=True)
        self.out_layer = nn.Sequential(
                nn.Linear(hidden_size, out_size),
        )

    def forward(self, xs):

        x_onehot = nn.functional.one_hot(xs, num_classes=self.in_size).float()

        _, (hidden, _) = self.lstm(x_onehot)
        hT = hidden[-1]
        out = self.out_layer(hT)

        return out
