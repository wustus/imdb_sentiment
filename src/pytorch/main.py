
# article: https://hassaanbinaslam.github.io/myblog/posts/2022-11-09-pytorch-lstm-imdb-sentiment-prediction.html
from imdb_dataset import IMDBDataset

import torch

ds = IMDBDataset("data/dataset.csv")

# research before use
# torch.nn.LSTM(input_size, hidden_size)
