
from torch import nn


class LSTMNetwork(nn.Module):

    def __init__(self, in_size, hidden_size, out_size, vocab_size):

        super().__init__()

        self.in_size = in_size
        self.embed = nn.Embedding(vocab_size, in_size)
        self.lstm = nn.LSTM(in_size, hidden_size, batch_first=True)
        self.out_layer = nn.Sequential(
                nn.Linear(hidden_size, out_size),
        )

    def forward(self, xs):

        xs = self.embed(xs)

        _, (hidden, _) = self.lstm(xs)
        hT = hidden[-1]
        out = self.out_layer(hT)

        return out
