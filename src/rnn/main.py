
from imdb_dataset import IMDBDataset
from network import Network

import numpy as np

ds = IMDBDataset("data/dataset.csv")
net = Network(ds.seq_len, 100, 2, ds.vocab)

print(len(ds.vocab))

epochs = 30

mWxh, mWhh, mWhy = np.zeros_like(net.Wxh), np.zeros_like(net.Whh), np.zeros_like(net.Why)
mbh, mby = np.zeros_like(net.bh), np.zeros_like(net.by)

# okay so ths is just shit i guess
#  -> we get about 50% correct with little learning
# might be doing something wrong but lets try lstm instead

for e in range(1, epochs+1):

    print(f"\rTraining Epoch {e}.")

    t_loss = 0

    for i, (d, l) in enumerate(zip(ds.training_data, ds.training_labels)):
        hprev = np.zeros((net.hidden_size, 1,))

        d_enc = [net.char_to_ix[c] for c in d]
        loss, h_prev, dWxh, dWhh, dWhy, dbh, dby = net.train(d_enc, l, hprev)
        t_loss += loss

        for param, dparam, mem in zip([net.Wxh, net.Whh, net.Why, net.bh, net.by], 
                                [dWxh, dWhh, dWhy, dbh, dby], 
                                [mWxh, mWhh, mWhy, mbh, mby]):
            mem +=  dparam**2
            param += -net.eta * dparam / np.sqrt(mem + 1e-8)

        if i > 0 and i % 1000 == 0:
            print(f"\r\tSeen {i} samples.", end="", flush=True)

    print(f"\r\tSeen {len(ds.training_data)} samples.", flush=True)
    correct = 0

    for d, l in zip(ds.test_data, ds.test_labels):
        hprev = np.zeros((net.hidden_size, 1,))
        d_enc = [net.char_to_ix[c] for c in d]
        p, h_prev = net(d_enc, hprev)
        correct += 1 if p.argmax() == l.argmax() else 0

    print(f"Epoch {e}: {correct} / {len(ds.test_data)}.")
