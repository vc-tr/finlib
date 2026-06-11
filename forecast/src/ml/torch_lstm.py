"""
A small, genuinely-trained LSTM direction model (optional dependency: torch).

This is a *real* sequence model, not a stub: it standardizes features on the
training window, builds causal sliding windows, and trains an LSTM classifier
with Adam, gradient clipping, a ReduceLROnPlateau learning-rate schedule, and
early stopping on a chronological validation tail. It implements the same
``DirectionModel`` protocol as the scikit-learn models, so the walk-forward
driver treats it identically and the same anti-lookahead guarantees apply.

torch is imported lazily so the rest of the forecast platform installs and runs
without it. Install with ``pip install -r requirements-ml.txt``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def torch_available() -> bool:
    """True if PyTorch can be imported."""
    try:
        import torch  # noqa: F401
        return True
    except Exception:
        return False


def _build_windows(
    X: np.ndarray, positions: np.ndarray, seq_len: int
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build causal sliding windows ending at each position.

    Window for position ``i`` is rows ``[i-seq_len+1 .. i]`` (all <= i), so a
    window never contains information from after the bar it describes.

    Returns:
        (windows, kept_positions) where windows has shape
        (n_kept, seq_len, n_features) and kept_positions are the positions with
        enough trailing history.
    """
    keep = positions[positions >= seq_len - 1]
    if len(keep) == 0:
        return np.empty((0, seq_len, X.shape[1]), dtype=np.float32), keep
    idx = keep[:, None] - np.arange(seq_len - 1, -1, -1)[None, :]
    return X[idx].astype(np.float32), keep


class LSTMDirectionModel:
    """LSTM binary direction classifier (P(up)) for the walk-forward driver."""

    def __init__(
        self,
        seq_len: int = 10,
        hidden_size: int = 16,
        num_layers: int = 1,
        epochs: int = 60,
        batch_size: int = 64,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        grad_clip: float = 1.0,
        patience: int = 8,
        val_frac: float = 0.2,
        seed: int = 0,
        device: str = "cpu",
    ):
        if not torch_available():
            raise ImportError(
                "LSTMDirectionModel requires PyTorch. "
                "Install it with: pip install -r requirements-ml.txt"
            )
        self.seq_len = seq_len
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.grad_clip = grad_clip
        self.patience = patience
        self.val_frac = val_frac
        self.seed = seed
        self.device = device

        self._mean: np.ndarray | None = None
        self._std: np.ndarray | None = None
        self._net = None

    # -- model definition -----------------------------------------------------
    def _make_net(self, n_features: int):
        import torch.nn as nn

        class _Net(nn.Module):
            def __init__(self, n_in, hidden, layers):
                super().__init__()
                self.lstm = nn.LSTM(n_in, hidden, layers, batch_first=True)
                self.head = nn.Linear(hidden, 1)

            def forward(self, x):
                out, _ = self.lstm(x)
                return self.head(out[:, -1, :]).squeeze(-1)  # logit per sequence

        return _Net(n_features, self.hidden_size, self.num_layers)

    # -- DirectionModel protocol ---------------------------------------------
    def fit(self, features: pd.DataFrame, labels: pd.Series, train_pos: np.ndarray) -> None:
        import torch

        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        gen = torch.Generator().manual_seed(self.seed)

        X_all = features.to_numpy()
        y_all = (labels.to_numpy() > 0.5).astype(np.float32)

        # Standardize on training rows only.
        train_rows = X_all[train_pos]
        self._mean = train_rows.mean(axis=0)
        self._std = train_rows.std(axis=0) + 1e-8
        X_std = (X_all - self._mean) / self._std

        windows, kept = _build_windows(X_std, np.sort(train_pos), self.seq_len)
        if len(kept) < self.seq_len + 2:
            self._net = None  # not enough data; predict() will fall back to 0.5
            return
        y = y_all[kept]

        # Chronological train/val split (no shuffling across the boundary).
        n = len(kept)
        n_val = max(1, int(n * self.val_frac))
        n_tr = n - n_val
        if n_tr < 1:
            n_tr, n_val = n, 0

        device = torch.device(self.device)
        Xt = torch.from_numpy(windows).to(device)
        yt = torch.from_numpy(y).to(device)
        X_tr, y_tr = Xt[:n_tr], yt[:n_tr]
        X_val, y_val = Xt[n_tr:], yt[n_tr:]

        net = self._make_net(X_all.shape[1]).to(device)
        opt = torch.optim.Adam(net.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=3)
        loss_fn = torch.nn.BCEWithLogitsLoss()

        best_val = float("inf")
        best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
        bad = 0

        for _ in range(self.epochs):
            net.train()
            perm = torch.randperm(n_tr, generator=gen)
            for s in range(0, n_tr, self.batch_size):
                b = perm[s : s + self.batch_size]
                opt.zero_grad()
                logits = net(X_tr[b])
                loss = loss_fn(logits, y_tr[b])
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), self.grad_clip)
                opt.step()

            # Validation + early stopping.
            net.eval()
            with torch.no_grad():
                if n_val > 0:
                    vloss = loss_fn(net(X_val), y_val).item()
                else:
                    vloss = loss_fn(net(X_tr), y_tr).item()
            sched.step(vloss)
            if vloss < best_val - 1e-5:
                best_val = vloss
                best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
                bad = 0
            else:
                bad += 1
                if bad >= self.patience:
                    break

        net.load_state_dict(best_state)
        net.eval()
        self._net = net

    def predict(self, features: pd.DataFrame, test_pos: np.ndarray) -> np.ndarray:
        import torch

        out = np.full(len(test_pos), 0.5, dtype=float)  # default: no edge
        if self._net is None:
            return out

        X_std = (features.to_numpy() - self._mean) / self._std
        windows, kept = _build_windows(X_std, np.asarray(test_pos), self.seq_len)
        if len(kept) == 0:
            return out

        device = torch.device(self.device)
        with torch.no_grad():
            logits = self._net(torch.from_numpy(windows).to(device))
            proba = torch.sigmoid(logits).cpu().numpy()

        # Map probabilities back onto the requested test_pos order.
        pos_to_proba = {int(p): float(pr) for p, pr in zip(kept, proba)}
        for j, p in enumerate(test_pos):
            if int(p) in pos_to_proba:
                out[j] = pos_to_proba[int(p)]
        return out
