
import numpy as np

class Network:

    def __init__(self, in_size, hidden_size, out_size, vocab):

        self.in_size = in_size
        self.hidden_size = hidden_size
        self.out_size = out_size

        self.vocab = vocab

        self.vocab_size = len(vocab)
        self.char_to_ix = { c: i for i,c in enumerate(vocab) }

        self.Wxh = np.random.randn(hidden_size, self.vocab_size)
        self.Whh = np.random.randn(hidden_size, hidden_size)
        self.Why = np.random.randn(out_size, hidden_size)

        self.bh = np.zeros((hidden_size, 1,))
        self.by = np.zeros((out_size, 1,))

        self.eta = 1e-1

    
    def __call__(self, ins, hprev):

        hs = {}
        hs[-1] = np.copy(hprev)

        for t in range(len(ins)):
            x = np.zeros((self.vocab_size, 1,))
            x[ins[t]] = 1
            hs[t] = np.tanh(np.dot(self.Wxh, x) + np.dot(self.Whh, hs[t-1]) + self.bh)

        last_h = hs[len(ins)-1]

        y = np.dot(self.Why, last_h) + self.by
        p = np.exp(y) / np.sum(np.exp(y))

        return p


    def train(self, inputs, target, hprev):

        xs, hs = {}, {}
        hs[-1] = np.copy(hprev)

        for t in range(len(inputs)):
            xs[t] = np.zeros((self.vocab_size, 1,))
            xs[t][inputs[t]] = 1
            hs[t] = np.tanh(np.dot(self.Wxh, xs[t]) + np.dot(self.Whh, hs[t-1]) + self.bh)

        last_h = hs[len(inputs)-1]

        y = np.dot(self.Why, last_h) + self.by
        p = np.exp(y) / np.sum(np.exp(y))

        loss = -np.sum(target * np.log(p))

        dWxh, dWhh, dWhy = np.zeros_like(self.Wxh), np.zeros_like(self.Whh), np.zeros_like(self.Why)
        dbh, dby = np.zeros_like(self.bh), np.zeros_like(self.by)

        dy = np.copy(p)
        dy -= target

        dWhy += np.dot(dy, last_h.T)
        dby += dy

        dh = np.dot(self.Why.T, dy)
        dhn = np.zeros_like(dh)

        for t in range(len(inputs)-2, 0):
            dh = dh + dhn
            dhraw = (1 - hs[t] * hs[t]) * dh

            dbh += dhraw
            dWxh += np.dot(dhraw, xs[t].T)
            dWhh += np.dot(dhraw, hs[t-1].T)
            dhn = np.dot(self.Whh.T, dhraw)

        self.Wxh -= self.eta * dWxh
        self.Whh -= self.eta * dWhh
        self.Why -= self.eta * dWhy

        self.bh -= self.eta * dbh
        self.by -= self.eta * dby

        return loss, hs[len(inputs)-1]
