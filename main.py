

def train_rnn(data_cutoff=None, sample_cutoff=None):

    from imdb_sentiment.utils.dataset_loader import DatasetLoader
    from imdb_sentiment.rnn.network import Network

    import numpy as np

    print("Training Recurrent Neural Network.")
    print(f"Data Cutoff: {data_cutoff}")
    print(f"Sample Cutoff: {sample_cutoff}")

    ds = DatasetLoader("data/dataset.csv", data_cutoff, sample_cutoff)
    net = Network(ds.vocab, 100, 2)

    print(f"Vocabulary size: {len(ds.vocab)}")

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

            for param, dparam, mem in zip(
                                        [net.Wxh, net.Whh, net.Why, net.bh, net.by],
                                        [dWxh, dWhh, dWhy, dbh, dby],
                                        [mWxh, mWhh, mWhy, mbh, mby]):

                mem += dparam**2
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


def train_lstm(data_cutoff=None, sample_cutoff=None):

    from imdb_sentiment.utils.dataset_loader import DatasetLoader
    from imdb_sentiment.lstm.network import Network

    import numpy as np

    print("Training Recurrent Neural Network.")
    print(f"Data Cutoff: {data_cutoff}")
    print(f"Sample Cutoff: {sample_cutoff}")

    ds = DatasetLoader("data/dataset.csv", data_cutoff, sample_cutoff)
    net = Network(ds.vocab, 200, 2)

    print(f"Vocabulary size: {len(ds.vocab)}")

    epochs = 10_000

    for e in range(1, epochs+1):

        indices = [i for i in range(len(ds.training_data))]
        np.random.shuffle(indices)

        c = 0

        for i in indices:
            d, l = ds.training_data[i], ds.training_labels[i]
            h_prev = np.zeros((net.hidden_size, 1,))
            c_prev = np.zeros((net.hidden_size, 1,))

            d_enc = [net.char_to_ix[c] for c in d]
            loss = net.train(d_enc, l, h_prev, c_prev, eta=1e-2)
            c += 1

            print(f"\r\tSeen {c} samples.", end="", flush=True)

        print(f"\r\tSeen {len(ds.training_data)} samples.", flush=True)
        correct = 0
        seen = 0

        t_loss = 0
        for d, l in zip(ds.test_data, ds.test_labels):
            h_prev = np.zeros((net.hidden_size, 1,))
            c_prev = np.zeros((net.hidden_size, 1,))
            d_enc = [net.char_to_ix[c] for c in d]
            p = net(d_enc, h_prev, c_prev)
            loss = -np.sum(l * np.log(p))
            t_loss += loss
            correct += 1 if p.argmax() == l.argmax() else 0
            seen += 1
            print(f"\rEpoch {e}: {correct} / {seen} ({(correct / seen) * 100: .2f}%).", end="", flush=True)

        print(f"\rEpoch {e}: {correct} / {len(ds.test_data)} ({(correct / seen) * 100: .2f}%) - Loss: {t_loss / len(ds.test_data): .4f}.", flush=True)


# article: https://hassaanbinaslam.github.io/myblog/posts/2022-11-09-pytorch-lstm-imdb-sentiment-prediction.html
def train_pytorch_lstm():

    from imdb_sentiment.pytorch.imdb_dataset import IMDBDataset
    from imdb_sentiment.pytorch.network import LSTMNetwork

    import torch
    from torch import optim
    from torch.utils.data import DataLoader

    ds = IMDBDataset("data/dataset.csv")
    dl = DataLoader(ds, batch_size=32, shuffle=True)

    vocab_size = ds.vocab_size

    device = torch.device("mps")

    net = LSTMNetwork(50, 1000, 2, vocab_size).to(device)

    opt = optim.Adagrad(net.parameters(), lr=1e-3)

    epochs = 200

    for e in range(1, epochs+1):

        net.train()

        for xs, ys, _, _ in dl:
            xs = xs.to(device)
            ys = ys.to(device)
            opt.zero_grad()
            out = net(xs)
            loss = torch.nn.functional.cross_entropy(out, ys.argmax(dim=1))
            loss.backward()
            opt.step()

        net.eval()

        correct = 0

        for _, _, xs, ys in dl:
            xs = xs.to(device)
            ys = ys.to(device)
            out = net(xs)
            print(out, ys)
            pred = out.argmax(dim=1)
            target = ys.argmax(dim=1)
            correct += (pred == target).sum().item()

        print(f"\rEpoch {e}: {correct} / {len(ds.training_data)} ({(correct / len(ds.training_data)) * 100: .2f}%).", flush=True)


if __name__ == "__main__":
    # train_lstm(None, 100)
    train_pytorch_lstm()
