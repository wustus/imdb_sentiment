
import numpy as np

class Network:

    def __init__(self, in_size, hidden_size, out_size, vocab):

        self.in_size = in_size
        self.hidden_size = hidden_size
        self.out_size = out_size

        self.vocab = vocab

        self.vocab_size = len(vocab)
        self.char_to_ix = { c: i for i,c in enumerate(vocab) }

        self.Wxh = np.random.randn(hidden_size, self.vocab_size) * 0.01
        self.Whh = np.random.randn(hidden_size, hidden_size) * 0.01
        self.Why = np.random.randn(out_size, hidden_size) * 0.01

        self.bh = np.zeros((hidden_size, 1,))
        self.by = np.zeros((out_size, 1,))

        self.eta = 5e-4

    
    def __call__(self, ins, hprev):

        hs = {}
        hs[-1] = np.copy(hprev)

        for t in range(len(ins)):
            x = np.zeros((self.vocab_size, 1,))
            x[ins[t]] = 1
            hs[t] = np.tanh(np.dot(self.Wxh, x) + np.dot(self.Whh, hs[t-1]) + self.bh)

        last_h = hs[len(ins)-1]

        y = np.dot(self.Why, last_h) + self.by
        y_stable = y - np.max(y)
        p = np.exp(y_stable) / np.sum(np.exp(y_stable))

        return p, last_h


    def train(self, inputs, target, hprev):

        xs, hs = {}, {}
        hs[-1] = np.copy(hprev)

        for t in range(len(inputs)):
            xs[t] = np.zeros((self.vocab_size, 1,))
            xs[t][inputs[t]] = 1
            hs[t] = np.tanh(np.dot(self.Wxh, xs[t]) + np.dot(self.Whh, hs[t-1]) + self.bh)

        last_h = hs[len(inputs)-1]

        y = np.dot(self.Why, last_h) + self.by
        y_stable = y - np.max(y)
        p = np.exp(y_stable) / np.sum(np.exp(y_stable))

        loss = -np.sum(target * np.log(p))

        dWxh, dWhh, dWhy = np.zeros_like(self.Wxh), np.zeros_like(self.Whh), np.zeros_like(self.Why)
        dbh, dby = np.zeros_like(self.bh), np.zeros_like(self.by)

        dy = np.copy(p)
        dy -= target

        dWhy += np.dot(dy, last_h.T)
        dby += dy

        dh = np.dot(self.Why.T, dy)
        dhn = np.zeros_like(dh)

        for t in reversed(range(0, len(inputs))):
            dh = dh + dhn
            dhraw = (1 - hs[t] * hs[t]) * dh

            dbh += dhraw
            dWxh += np.dot(dhraw, xs[t].T)
            dWhh += np.dot(dhraw, hs[t-1].T)
            dhn = np.dot(self.Whh.T, dhraw)

        for grad in [dWxh, dWhh, dWhy, dbh, dby]:
            np.clip(grad, -5, 5, out=grad)

        return loss, last_h, dWxh, dWhh, dWhy, dbh, dby
