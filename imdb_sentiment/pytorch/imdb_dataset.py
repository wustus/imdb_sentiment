
from torch.utils.data import Dataset
from nltk.tokenize import word_tokenize
from torchtext.vocab import GloVe

import torch
import nltk
import re


class IMDBDataset(Dataset):

    def __init__(self, path, dim=200, cutoff=None, test=False):
        self.data = []
        self.labels = []

        self.vocab = set()
        self.vocab_size = 0
        self.seq_len = 0

        self.stopwords = nltk.corpus.stopwords.words("english")

        with open(path) as f:

            lines = f.readlines()[1:]

            if cutoff:
                lines = lines[:cutoff]

            train_cutoff = int(len(lines) / 2)
            if test:
                lines = lines[train_cutoff:]
            else:
                lines = lines[:train_cutoff]

            d_size = len(lines)

            for i, line in enumerate(lines):
                line = line.split(",")
                l = line[-1]
                l = torch.Tensor([1]) if l.strip() == "positive" else torch.Tensor([0])
                d = ",".join(line[:-1])
                d = self.preprocess(d)

                print(f"\rProcessed {i} / {d_size} {"test" if test else "training"} entries.", end="", flush=True)

                self.data.append(d)
                self.labels.append(l)

            print(f"\rProcessed {d_size} / {d_size} {"test" if test else "training"} entries.", flush=True)
             
        self.vocab = list(self.vocab)
        self.vocab += ["="]
        self.vocab_size = len(self.vocab)
        self.glove = GloVe(name="6B", dim=200)

        for i in range(len(self.data)):
            self.data[i] = torch.stack([
                self.glove.get_vecs_by_tokens(token, lower_case_backup=True)
                if token in self.glove.stoi else torch.zeros(200)
                for token in self.data[i]
            ])

    def preprocess(self, t):
        t = re.sub(r"<.*?>", "", t)
        t = word_tokenize(t)
        # remove non-alphabetic characters
        # test with other characters
        #   t = [w for w in t if w.isalpha()]
        #   t = [w.lower() for w in t if w.lower() not in self.stopwords]

        self.vocab = self.vocab.union(set(t))

        if len(t) > self.seq_len:
            self.seq_len = len(t)
        
        return t


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx], len(self.data[idx])
