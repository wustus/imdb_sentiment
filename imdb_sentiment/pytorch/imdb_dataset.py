
from torch.utils.data import Dataset
from nltk.tokenize import word_tokenize

import torch
import nltk
import re


class IMDBDataset(Dataset):

    def __init__(self, path):
        self.training_data = []
        self.training_labels = []

        self.test_data = []
        self.test_labels = []

        self.vocab = set()
        self.vocab_size = 0
        self.seq_len = 0

        self.stopwords = nltk.corpus.stopwords.words("english")

        with open(path) as f:

            lines = f.readlines()[1:]
            lines = lines[:1000]
            train_cutoff = len(lines) / 2

            d_size = len(lines)

            for i, line in enumerate(lines):
                line = line.split(",")
                l = line[-1]
                l = torch.Tensor([1, 0]) if l.strip() == "positive" else torch.Tensor([0, 1])
                d = ",".join(line[:-1])
                d = self.preprocess(d)

                print(f"\rProcessed {i} / {d_size} entries.", end="", flush=True)

                if i < train_cutoff:
                    self.training_data.append(d)
                    self.training_labels.append(l)
                else:
                    self.test_data.append(d)
                    self.test_labels.append(l)
            print(f"\rProcessed {d_size} / {d_size} entries.", flush=True)
             
        self.vocab = list(self.vocab)
        self.vocab += ["="]
        self.vocab_size = len(self.vocab)

        self.char_to_ix = { c: i for i, c in enumerate(self.vocab) }

        self.pad_data()

        for i in range(len(self.training_data)):
            self.training_data[i] = torch.tensor([self.char_to_ix[w] for w in self.training_data[i]], dtype=torch.long)

        for i in range(len(self.test_data)):
            self.test_data[i] = torch.tensor([self.char_to_ix[w] for w in self.test_data[i]], dtype=torch.long)

    def preprocess(self, t):
        t = re.sub(r"<.*?>", "", t)
        t = word_tokenize(t)
        t = [w for w in t if w.isalpha()]
        t = [w.lower() for w in t if w.lower() not in self.stopwords]

        self.vocab = self.vocab.union(set(t))

        if len(t) > self.seq_len:
            self.seq_len = len(t)
        
        return t

    def pad_data(self):
        for i, d in enumerate(self.training_data):
            self.training_data[i] = d + ["="] * (self.seq_len - len(d))

        for i, d in enumerate(self.test_data):
            self.test_data[i] = d + ["="] * (self.seq_len - len(d))

    def __len__(self):
        return len(self.training_data)

    def __getitem__(self, idx):
        return self.training_data[idx], self.training_labels[idx], self.test_data[idx], self.test_labels[idx]
