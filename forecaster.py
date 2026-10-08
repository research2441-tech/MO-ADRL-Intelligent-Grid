"""
Temporal Transformer forecaster, Eq. (2)-(3).

Input  X_t  in R^{n x d}: the last n samples of [load, PV, wind, irradiance,
wind speed, temperature, sin(hour), cos(hour)], each min-max normalised.
Output [L_hat_{t+1}, P_hat_PV,t+1, P_hat_W,t+1]: one-interval-ahead forecasts.
"""
import math
import numpy as np
import torch
import torch.nn as nn

FEATURES = ["load", "pv", "wind", "irradiance", "wind_speed", "temperature"]


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=2048):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class TemporalTransformer(nn.Module):
    def __init__(self, n_features=8, d_model=64, n_heads=4, n_layers=2, ff=128, dropout=0.1):
        super().__init__()
        self.inp = nn.Linear(n_features, d_model)
        self.pos = PositionalEncoding(d_model)
        layer = nn.TransformerEncoderLayer(d_model, n_heads, ff, dropout, batch_first=True)
        self.enc = nn.TransformerEncoder(layer, n_layers)
        self.head = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, 3))

    def forward(self, x):                      # x: (B, n, d)
        h = self.enc(self.pos(self.inp(x)))
        return self.head(h[:, -1])             # (B, 3)


class Forecaster:
    """Wraps training and one-step prediction of the Temporal Transformer."""

    def __init__(self, lookback=96, device="cpu"):
        self.n = lookback
        self.device = device
        self.model = TemporalTransformer().to(device)
        self.lo = None
        self.hi = None

    def _matrix(self, prof):
        cols = [prof[f] for f in FEATURES]
        m = np.stack(cols, 1).astype(np.float32)
        if self.lo is None:
            self.lo, self.hi = m.min(0), m.max(0) + 1e-6
        m = (m - self.lo) / (self.hi - self.lo)
        h = prof["hour"]
        tenc = np.stack([np.sin(2 * np.pi * h / 24), np.cos(2 * np.pi * h / 24)], 1).astype(np.float32)
        return np.concatenate([m, tenc], 1)

    def fit(self, prof, train_days=300, epochs=5, batch=256, lr=1e-3, max_samples=20000, seed=0):
        rng = np.random.default_rng(seed)
        m = self._matrix(prof)
        self._m = m
        last = train_days * 96 - 1
        idx = rng.choice(np.arange(self.n, last), size=min(max_samples, last - self.n), replace=False)
        opt = torch.optim.Adam(self.model.parameters(), lr=lr)
        lossf = nn.MSELoss()
        self.model.train()
        hist = []
        for ep in range(epochs):
            rng.shuffle(idx)
            tot = 0.0
            for b in range(0, len(idx), batch):
                ii = idx[b:b + batch]
                x = np.stack([m[i - self.n + 1:i + 1] for i in ii])
                y = m[ii + 1, :3]
                x = torch.from_numpy(x).to(self.device)
                y = torch.from_numpy(y).to(self.device)
                loss = lossf(self.model(x), y)
                opt.zero_grad()
                loss.backward()
                opt.step()
                tot += loss.item() * len(ii)
            hist.append(tot / len(idx))
        self.model.eval()
        return hist

    @torch.no_grad()
    def predict_next(self, prof, k):
        if not hasattr(self, "_m"):
            self._m = self._matrix(prof)
        k = max(k, self.n - 1)
        x = torch.from_numpy(self._m[k - self.n + 1:k + 1][None]).to(self.device)
        y = self.model(x).cpu().numpy()[0]
        y = np.clip(y, 0, 1) * (self.hi[:3] - self.lo[:3]) + self.lo[:3]
        return float(y[0]), float(y[1]), float(y[2])
