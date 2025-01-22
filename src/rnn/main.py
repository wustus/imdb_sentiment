
from imdb_dataset import IMDBDataset
from network import Network

import numpy as np

ds = IMDBDataset("data/dataset.csv")
net = Network(ds.seq_len, 100, 2, ds.vocab)

hprev = np.zeros((net.hidden_size, 1,))

epochs = 30

for e in range(1, epochs+1):

    t_loss = 0

    for d, l in zip(ds.data, ds.labels):

        d_enc = [net.char_to_ix[c] for c in d]
        loss, hprev = net.train(d_enc, l, hprev)
        t_loss += loss

    correct = 0

    for d, l in zip(ds.test, ds.test_labels):

        d_enc = [net.char_to_ix[c] for c in d]
        p = net(d_enc, hprev)
        correct += 1 if p.argmax() == l.argmax() else 0

    print(f"Epoch {e}: {correct} / {len(ds.test)}.")
