
import numpy as np


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def sigmoid_prime(x):
    s = sigmoid(x)
    return s * (1 - s)


class Network:

    def __init__(self, vocab, hidden_size, out_size):

        self.hidden_size = hidden_size
        self.out_size = out_size
        self.vocab = vocab

        self.vocab_size = len(vocab)
        self.char_to_ix = { c: i for i, c in enumerate(vocab) }

        # forget gate
        self.Wfx = np.random.randn(hidden_size, self.vocab_size) * 0.01
        self.Whf = np.random.randn(hidden_size, hidden_size) * 0.01
        self.bf = np.random.randn(hidden_size, 1) * 0.01

        # input gate
        self.Wix = np.random.randn(hidden_size, self.vocab_size) * 0.01
        self.Whi = np.random.randn(hidden_size, hidden_size) * 0.01
        self.bi = np.random.randn(hidden_size, 1) * 0.01

        # cell gate
        self.Wcx = np.random.randn(hidden_size, self.vocab_size) * 0.01
        self.Whc = np.random.randn(hidden_size, hidden_size) * 0.01
        self.bc = np.random.randn(hidden_size, 1) * 0.01

        # output gate
        self.Wox = np.random.randn(hidden_size, self.vocab_size) * 0.01
        self.Who = np.random.randn(hidden_size, hidden_size) * 0.01
        self.bo = np.random.randn(hidden_size, 1) * 0.01

        # output
        self.Why = np.random.randn(out_size, hidden_size) * 0.01
        self.by = np.random.randn(out_size, 1) * 0.01


    def __call__(self, ins, h_prev, c_prev):

        hs = {}
        cs = {}

        hs[-1] = h_prev
        cs[-1] = c_prev

        for t in range(len(ins)):
            x = np.zeros((self.vocab_size, 1))
            x[ins[t]] = 1.0
            
            zft = self.Wfx@x + self.Whf@h_prev + self.bf
            ft = sigmoid(zft)

            zit = self.Wix@x + self.Whi@h_prev + self.bi
            it = sigmoid(zit)

            zot = self.Wox@x + self.Who@h_prev + self.bo
            ot = sigmoid(zot)

            zct = self.Wcx@x + self.Whc@h_prev + self.bc
            cct = np.tanh(zct)

            ct = ft * cs[t-1] + it * cct
            ht = ot * np.tanh(ct)

            cs[t] = ct
            hs[t] = ht

        ht = hs[len(ins)-1]
        ct = cs[len(ins)-1]

        # y = V@ht + bp
        y = self.Why@ht + self.by

        return y


    def train(self, ins, target, h_prev, c_prev, eta=1e-2):

        xs = {}
        zfs, zis, zos, zcs = {}, {}, {}, {}
        _is = {}
        os = {}
        fs = {}
        ccs = {}
        hs = {}
        cs = {}

        hs[-1] = h_prev
        cs[-1] = c_prev

        """
         Feed forward through the network given previous hidden and cell weights.
         ft = sig(Wf @ xt + Uf @ h_prev + bf)
         it = sig(Wi @ xt + Ui @ h_prev + bi)
         ot = sig(Wo @ xt + Uo @ h_prev + bo)
         cct = tanh(Wc @ xt + Uc @ h_prev + bc)
         ct = ft * c_prev + it * cct
         ht = ot * tanh(ct)
        """
        for t in range(len(ins)):
            x = np.zeros((self.vocab_size, 1))
            x[ins[t]] = 1.0
            xs[t] = x
            
            zft = self.Wfx@x + self.Whf@h_prev + self.bf
            zfs[t] = zft
            ft = sigmoid(zft)
            fs[t] = ft

            zit = self.Wix@x + self.Whi@h_prev + self.bi
            zis[t] = zit
            it = sigmoid(zit)
            _is[t] = it

            zot = self.Wox@x + self.Who@h_prev + self.bo
            zos[t] = zot
            ot = sigmoid(zot)
            os[t] = ot

            zct = self.Wcx@x + self.Whc@h_prev + self.bc
            zcs[t] = zct
            cct = np.tanh(zct)
            ccs[t] = cct

            ct = ft * cs[t-1] + it * cct
            ht = ot * np.tanh(ct)

            cs[t] = ct
            hs[t] = ht

        ht = hs[len(ins)-1]
        ct = cs[len(ins)-1]

        # y = V@ht + bp
        y = self.Why@ht + self.by

        # L = 1/2 (y - target)^2
        # dL/dy = y - target
        dy = y - target

        # dL/dV = dL/dy * dy/dV = (y-target) * ht.T
        dWhy = dy@ht.T
        # dL/dby = dL/dy @ dy/dby = (y-target) * 1
        dby = dy

        # dL/dht = dL/dy ∆ dy/dHt = (y-target) * V
        dhn = self.Why.T @ dy
        dcn = np.zeros_like(cs[len(ins)-1])

        dWfx = np.zeros_like(self.Wfx)
        dWhf = np.zeros_like(self.Whf)
        dbf = np.zeros_like(self.bf)

        dWix = np.zeros_like(self.Wix)
        dWhi = np.zeros_like(self.Whi)
        dbi = np.zeros_like(self.bi)

        dWcx = np.zeros_like(self.Wcx)
        dWhc = np.zeros_like(self.Whc)
        dbc = np.zeros_like(self.bc)

        dWox = np.zeros_like(self.Wox)
        dWho = np.zeros_like(self.Who)
        dbo = np.zeros_like(self.bo)

        for t in reversed(range(0, len(ins))):

            dht = dhn

            # dL/dct = dL/dht * dht/dct
            t_ct = np.tanh(ct)
            dct = dcn + (dht * os[t] * (1 - t_ct**2))

            # dL/dot = dL/dht * dht/dot
            dot = dht * np.tanh(ct) * sigmoid_prime(zos[t])
            dft = dct * cs[t-1] * sigmoid_prime(zfs[t])
            dit = dct * ccs[t] * sigmoid_prime(zis[t])
            dcct = dct * _is[t] * (1 - np.tanh(zcs[t])**2)

            xt_t = xs[t].T
            h_prev_t = hs[t-1].T

            # forget gate
            dWfx += dft @ xt_t
            dWhf += dft @ h_prev_t
            dbf += dft

            # input gate
            dWix += dit @ xt_t
            dWhi += dit @ h_prev_t
            dbi += dit

            # candidate cell gate
            dWcx += dcct @ xt_t
            dWhc += dcct @ h_prev_t
            dbc += dcct

            # output gate
            dWox += dot @ xt_t
            dWho += dot @ h_prev_t
            dbo += dot

            dh_prev = (self.Whf.T @ dft) + (self.Whi.T @ dit) + (self.Whc.T @ dcct) + (self.Who.T @ dot)
            dc_prev = dct * fs[t]

            dhn = dh_prev
            dcn = dc_prev


        for grad in [dhn, dcn, dWfx, dWhf, dbf, dWix, dWhi, dbi, dWcx, dWhc, dbc, dWox, dWho, dbo, dWhy, dby]:
            np.clip(grad, -5, 5, out=grad)

        self.Wfx -= eta * dWfx
        self.Whf -= eta * dWhf
        self.bf -= eta * dbf
         
        self.Wix -= eta * dWix
        self.Whi -= eta * dWhi
        self.bi -= eta * dbi
         
        self.Wcx -= eta * dWcx
        self.Whc -= eta * dWhc
        self.bc -= eta * dbc
         
        self.Wox -= eta * dWox
        self.Who -= eta * dWho
        self.bo -= eta * dbo

        self.Why -= eta * dWhy
        self.by -= eta * dby
