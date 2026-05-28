"""
lstm_model.py
Lightweight LSTM implementation using numpy (no PyTorch/TensorFlow required).
Used for sequence-based demand forecasting per SKU.
"""

import numpy as np
import pickle
from pathlib import Path


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


def tanh(x):
    return np.tanh(np.clip(x, -500, 500))


class LSTMCell:
    """Single LSTM cell with learnable weights."""

    def __init__(self, input_size: int, hidden_size: int, seed: int = 0):
        rng = np.random.RandomState(seed)
        scale = 0.1
        sz = hidden_size

        # Combined weight matrix: [W_i, W_f, W_g, W_o]  shape (4*sz, input_size + sz)
        self.W = rng.randn(4 * sz, input_size + sz) * scale
        self.b = np.zeros(4 * sz)
        self.hidden_size = sz

    def forward(self, x: np.ndarray, h_prev: np.ndarray, c_prev: np.ndarray):
        combined = np.concatenate([x, h_prev])
        gates = self.W @ combined + self.b

        sz = self.hidden_size
        i = sigmoid(gates[0:sz])
        f = sigmoid(gates[sz:2*sz])
        g = tanh(gates[2*sz:3*sz])
        o = sigmoid(gates[3*sz:4*sz])

        c = f * c_prev + i * g
        h = o * tanh(c)
        return h, c


class SimpleLSTM:
    """
    Univariate LSTM forecaster.
    Trains via truncated BPTT with SGD + momentum.
    """

    def __init__(self, seq_len: int = 28, hidden_size: int = 32,
                 lr: float = 0.001, epochs: int = 30, seed: int = 42):
        self.seq_len = seq_len
        self.hidden_size = hidden_size
        self.lr = lr
        self.epochs = epochs
        self.seed = seed
        self.cell = None
        self.W_out = None
        self.b_out = None
        self.mean_ = 0.0
        self.std_ = 1.0

    def _make_sequences(self, series: np.ndarray):
        X, y = [], []
        for i in range(len(series) - self.seq_len):
            X.append(series[i: i + self.seq_len])
            y.append(series[i + self.seq_len])
        return np.array(X), np.array(y)

    def fit(self, series: np.ndarray):
        # Normalize
        self.mean_ = series.mean()
        self.std_  = series.std() + 1e-8
        s = (series - self.mean_) / self.std_

        X, y = self._make_sequences(s)
        if len(X) == 0:
            return self

        rng = np.random.RandomState(self.seed)
        self.cell  = LSTMCell(1, self.hidden_size, seed=self.seed)
        self.W_out = rng.randn(self.hidden_size) * 0.1
        self.b_out = 0.0

        for epoch in range(self.epochs):
            idx = rng.permutation(len(X))
            total_loss = 0.0
            for i in idx:
                h = np.zeros(self.hidden_size)
                c = np.zeros(self.hidden_size)
                hs = []
                for t in range(self.seq_len):
                    x_t = np.array([X[i, t]])
                    h, c = self.cell.forward(x_t, h, c)
                    hs.append(h.copy())

                pred = self.W_out @ h + self.b_out
                err  = pred - y[i]
                total_loss += err ** 2

                # Output layer gradient
                d_W_out = err * h
                d_b_out = err
                d_h = err * self.W_out

                # Crude BPTT: update weights with gradient approx
                self.W_out -= self.lr * d_W_out
                self.b_out -= self.lr * d_b_out

                # Cell weight gradient (last step only — simplified BPTT)
                sz = self.hidden_size
                x_t = np.array([X[i, -1]])
                combined = np.concatenate([x_t, hs[-2] if len(hs) > 1 else np.zeros(sz)])
                # d_h has shape (sz,); combined has shape (input+sz,)
                # W has shape (4*sz, input+sz); gradient must match
                d_h_full = np.tile(d_h, 4)          # (4*sz,)
                d_W = np.outer(d_h_full, combined) * 0.01   # (4*sz, input+sz)
                self.cell.W -= self.lr * np.clip(d_W, -1, 1)

        return self

    def predict_next(self, series: np.ndarray, steps: int = 28) -> np.ndarray:
        s = (series - self.mean_) / self.std_
        window = list(s[-self.seq_len:])
        preds = []

        for _ in range(steps):
            h = np.zeros(self.hidden_size)
            c = np.zeros(self.hidden_size)
            for val in window:
                x_t = np.array([val])
                h, c = self.cell.forward(x_t, h, c)
            p = self.W_out @ h + self.b_out
            preds.append(p)
            window = window[1:] + [p]

        return np.array(preds) * self.std_ + self.mean_

    def save(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str) -> "SimpleLSTM":
        with open(path, "rb") as f:
            return pickle.load(f)
