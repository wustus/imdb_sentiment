
import numpy as np

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


class Network:

    def __init__(self, in_size, hidden_size, out_size, vocab):

        self.in_size = in_size
        self.hidden_size = hidden_size
        self.out_size = out_size
        self.vocab = vocab

        self.vocab_size = len(vocab)

        # forget gate
        self.Wfx = np.random.randn(hidden_size, in_size)
        self.Wff = np.random.randn(hidden_size, hidden_size)
        self.bf = np.random.randn(hidden_size, 1)

        # input gate
        self.Wix = np.random.randn(hidden_size, in_size)
        self.Wii = np.random.randn(hidden_size, hidden_size)
        self.bi = np.random.randn(hidden_size, 1)

        # cell gate
        self.Wcx = np.random.randn(hidden_size, in_size)
        self.Wcc = np.random.randn(hidden_size, hidden_size)
        self.bc = np.random.randn(hidden_size, 1)

        # output gate
        self.Wox = np.random.randn(hidden_size, in_size)
        self.Woo = np.random.randn(hidden_size, hidden_size)
        self.bo = np.random.randn(hidden_size, 1)

        # output
        self.Woy = np.random.randn(out_size, hidden_size)
        self.by = np.random.randn(out_size, 1)



    """
     Feed forward through the network given previous hidden and cell weights.
     ft = sig(Wf * xt + Uf * h_prev + bf)
     it = sig(Wi * xt + Ui * h_prev + bi)
     ot = sig(Wo * xt + Uo * h_prev + bo)
     cct = tanh(Wc * xt + Uc * h_prev + bc)
     ct = ft O c_prev + it O cct                # O - hadamard product
     ht = ot O tanh(ct)                         # O - hadamard product
    """
    def feed_forward(self, ins, h_prev, c_prev):

        xs = {}
        zfs, zis, zos, zcs = {}, {}, {}, {}
        hs = {}
        cs = {}

        hs[-1] = h_prev
        cs[-1] = c_prev

        for t in range(len(ins)):
            x = np.zeros((self.vocab_size, 1))
            x[ins[t]] = 1.0
            xs[t] = x
            
            zft = np.dot(self.Wfx, x) + np.dot(self.Wfh, h_prev) + self.bf
            zfs[t] = zft
            ft = sigmoid(zft)

            zit = np.dot(self.Wix, x) + np.dot(self.Wih, h_prev) + self.bi
            zis[t] = zit
            it = sigmoid(zit)

            zot = np.dot(self.Wox, x) + np.dot(self.Woh, h_prev) + self.bo
            zos[t] = zot
            ot = sigmoid(zot)

            zct = np.dot(self.Wcx, x) + np.dot(self.Wch, h_prev) + self.bc
            zcs[t] = zct
            cct = np.tanh(zct)

            ct = ft * cs[t-1] + it * cct
            ht = ot * np.tanh(ct)

            cs[t] = ct
            hs[t] = ht
