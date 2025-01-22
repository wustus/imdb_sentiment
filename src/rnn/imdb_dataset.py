
from nltk.tokenize import word_tokenize

import numpy as np
import nltk
import re

class IMDBDataset():

    def __init__(self, path):

        self.data = []
        self.labels = []

        self.test = []
        self.test_labels = []

        self.vocab = set()
        self.seq_len = 0

        self.stopwords = nltk.corpus.stopwords.words("english")

        with open(path, encoding="utf-8") as f:
            lines = f.readlines()[1:][:1000]
            train_cutoff = len(lines) / 2
            d_size = len(lines)
            for i, line in enumerate(lines):
                print(f"\rProcessed {i} / {d_size} entries.", end="", flush=True)
                line = line.split(",")
                l = line[-1]
                l = np.array([[1],[0]]) if l == "positive" else np.array([[0],[1]])
                d = ",".join(line[:-1])
                d = self.preprocess(d)

                if i < train_cutoff:
                    self.data.append(d)
                    self.labels.append(l)
                else:
                    self.test.append(d)
                    self.test_labels.append(l)
            print(f"\rProcessed {d_size} / {d_size} entries.", flush=True)

        self.vocab = list(self.vocab)
        self.vocab += ["="]

        self.pad_data()

    def preprocess(self, t):
        t = re.sub(r"<.*?>", "", t)
        t = word_tokenize(t)
        t = [w for w in t if w.isalpha()]
        t = [w.lower() for w in t if w.lower() not in self.stopwords]
        t = " ".join(t)

        self.vocab = self.vocab.union(set(t))

        if len(t) > self.seq_len:
            self.seq_len = len(t)
        
        return t

    def pad_data(self):
        for i, d in enumerate(self.data):
            self.data[i] = d + "=" * (self.seq_len - len(d))

        for i, d in enumerate(self.test):
            self.test[i] = d + "=" * (self.seq_len - len(d))
