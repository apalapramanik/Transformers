import torch


class CharDataset:
    """
    PURPOSE:
    Character-level dataset for language modeling.

    Given a long text sequence, it produces:
    - input sequence of length T
    - target sequence shifted by 1
    """

    def __init__(self, text, seq_len, vocab=None, stride=None):
        self.seq_len = seq_len
        self.stride = stride if stride is not None else seq_len

        if vocab is not None:
            # Use the caller-supplied vocab so val/test indices match the model.
            # Characters absent from vocab fall back to the space index.
            self.stoi = vocab
            self.itos = {i: ch for ch, i in vocab.items()}
            self.vocab_size = len(vocab)
            unk_idx = vocab.get(" ", 0)
            indices = [vocab.get(c, unk_idx) for c in text]
        else:
            chars = sorted(list(set(text)))
            self.stoi = {ch: i for i, ch in enumerate(chars)}
            self.itos = {i: ch for ch, i in self.stoi.items()}
            self.vocab_size = len(chars)
            indices = [self.stoi[c] for c in text]

        # Encode entire text as integers
        self.data = torch.tensor(indices, dtype=torch.long)

    def __len__(self):
        return (len(self.data) - self.seq_len) // self.stride

    def __getitem__(self, idx):
        start = idx * self.stride
        x = self.data[start : start + self.seq_len]
        y = self.data[start + 1 : start + self.seq_len + 1]
        return x, y
