
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
        self.Wfx = np.random.randn(hidden_size, self.vocab_size) * np.sqrt(1 / self.vocab_size)
        self.Whf = np.random.randn(hidden_size, hidden_size) * np.sqrt(1 / self.hidden_size)
        self.bf = np.full((hidden_size, 1), 2, dtype=float)

        # input gate
        self.Wix = np.random.randn(hidden_size, self.vocab_size) * np.sqrt(1 / self.vocab_size)
        self.Whi = np.random.randn(hidden_size, hidden_size) * np.sqrt(1 / self.hidden_size)
        self.bi = np.zeros((hidden_size, 1))

        # cell gate
        self.Wcx = np.random.randn(hidden_size, self.vocab_size) * np.sqrt(1 / self.vocab_size)
        self.Whc = np.random.randn(hidden_size, hidden_size) * np.sqrt(1 / self.hidden_size)
        self.bc = np.zeros((hidden_size, 1))

        # output gate
        self.Wox = np.random.randn(hidden_size, self.vocab_size) * np.sqrt(1 / self.vocab_size)
        self.Who = np.random.randn(hidden_size, hidden_size) * np.sqrt(1 / self.hidden_size)
        self.bo = np.zeros((hidden_size, 1))

        # output
        self.Why = np.random.randn(out_size, hidden_size)
        self.by = np.random.randn(out_size, 1)

        self.mem = {}


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
        y = y - np.max(y)
        p = np.exp(y) / np.sum(np.exp(y), axis=0, keepdims=True)

        return p


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
        y = y - np.max(y)
        p = np.exp(y) / np.sum(np.exp(y), axis=0, keepdims=True)

        dy = p - target

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
            ct = cs[t]

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

        # forget gate
        if "Wfx" not in self.mem:
            self.mem["Wfx"] = np.zeros_like(self.Wfx)

        if "Whf" not in self.mem:
            self.mem["Whf"] = np.zeros_like(self.Whf)

        if "bf" not in self.mem:
            self.mem["bf"] = np.zeros_like(self.bf)

        self.mem["Wfx"] += dWfx**2
        self.Wfx -= eta * dWfx / (np.sqrt(self.mem["Wfx"]) + 1e-8)
        self.mem["Whf"] += dWhf**2
        self.Whf -= eta * dWhf / (np.sqrt(self.mem["Whf"]) + 1e-8)
        self.mem["bf"] += dbf**2
        self.bf -= eta * dbf / (np.sqrt(self.mem["bf"]) + 1e-8)
         
        # input gate
        if "Wix" not in self.mem:
            self.mem["Wix"] = np.zeros_like(self.Wix)
        if "Whi" not in self.mem:
            self.mem["Whi"] = np.zeros_like(self.Whi)
        if "bi" not in self.mem:
            self.mem["bi"] = np.zeros_like(self.bi)

        self.mem["Wix"] += dWix**2
        self.Wix -= eta * dWix / (np.sqrt(self.mem["Wix"]) + 1e-8)
        self.mem["Whi"] += dWhi**2
        self.Whi -= eta * dWhi / (np.sqrt(self.mem["Whi"]) + 1e-8)
        self.mem["bi"] += dbi**2
        self.bi -= eta * dbi / (np.sqrt(self.mem["bi"]) + 1e-8)

        # cell gate
        if "Wcx" not in self.mem:
            self.mem["Wcx"] = np.zeros_like(self.Wcx)
        if "Whc" not in self.mem:
            self.mem["Whc"] = np.zeros_like(self.Whc)
        if "bc" not in self.mem:
            self.mem["bc"] = np.zeros_like(self.bc)

        self.mem["Wcx"] += dWcx**2
        self.Wcx -= eta * dWcx / (np.sqrt(self.mem["Wcx"]) + 1e-8)
        self.mem["Whc"] += dWhc**2
        self.Whc -= eta * dWhc / (np.sqrt(self.mem["Whc"]) + 1e-8)
        self.mem["bc"] += dbc**2
        self.bc -= eta * dbc / (np.sqrt(self.mem["bc"]) + 1e-8)

        # output gate
        if "Wox" not in self.mem:
            self.mem["Wox"] = np.zeros_like(self.Wox)
        if "Who" not in self.mem:
            self.mem["Who"] = np.zeros_like(self.Who)
        if "bo" not in self.mem:
            self.mem["bo"] = np.zeros_like(self.bo)

        self.mem["Wox"] += dWox**2
        self.Wox -= eta * dWox / (np.sqrt(self.mem["Wox"]) + 1e-8)
        self.mem["Who"] += dWho**2
        self.Who -= eta * dWho / (np.sqrt(self.mem["Who"]) + 1e-8)
        self.mem["bo"] += dbo**2
        self.bo -= eta * dbo / (np.sqrt(self.mem["bo"]) + 1e-8)

        # output layer
        if "Why" not in self.mem:
            self.mem["Why"] = np.zeros_like(self.Why)
        if "by" not in self.mem:
            self.mem["by"] = np.zeros_like(self.by)

        self.mem["Why"] += dWhy**2
        self.Why -= eta * dWhy / (np.sqrt(self.mem["Why"]) + 1e-8)
        self.mem["by"] += dby**2
        self.by -= eta * dby / (np.sqrt(self.mem["by"]) + 1e-8)
