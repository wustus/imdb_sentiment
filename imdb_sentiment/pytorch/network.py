
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence


class LSTMNetwork(nn.Module):

    def __init__(self, in_size, hidden_size, out_size, vocab_size, dropout=0):

        super().__init__()

        self.in_size = in_size
        self.lstm = nn.LSTM(in_size, hidden_size, batch_first=True, dropout=dropout)
        self.out_layer = nn.Sequential(
                nn.Linear(hidden_size, out_size),
        )

    def forward(self, xs, lengths):

        xs = pack_padded_sequence(xs, lengths.cpu(), batch_first=True)
        _, (hidden, _) = self.lstm(xs)
        hT = hidden[-1, :, :]
        out = self.out_layer(hT)

        return out
