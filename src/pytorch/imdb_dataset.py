
from torch.utils.data import Dataset
from nltk.tokenize import word_tokenize

import nltk
import re

nltk.download("punkt_tab")

class IMDBDataset(Dataset):

    def __init__(self, path):
        self.data = []
        self.labels = []

        self.stopwords = nltk.corpus.stopwords.words("english")

        with open(path) as f:
            for line in f.readlines()[1:]:
                line = line.split(",")
                l = line[-1]
                d = ",".join(line[:-1])
                self.preprocess(d)
                self.data.append(d)
                self.labels.append(l)

    def preprocess(self, t):
        t = re.sub(r"<.*?>", "", t)
        t = word_tokenize(t)
        t = [w for w in t if w.isalpha()]
        t = [w.lower() for w in t if w.lower() not in self.stopwords]
        print(t)
        input()


    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]
