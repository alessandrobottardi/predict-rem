import math
import random
from dataclasses import dataclass
from typing import Dict, Any, Optional, List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, WeightedRandomSampler

from sklearn.metrics import f1_score, roc_auc_score, accuracy_score, precision_recall_curve

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


# --------------------------
# Utilities
# --------------------------

def _seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _to_tensor(x):
    if isinstance(x, np.ndarray):
        return torch.tensor(x, dtype=torch.float32)
    elif torch.is_tensor(x):
        return x.float()
    else:
        raise TypeError("Expected numpy array or torch tensor")


def _to_long(y):
    if isinstance(y, np.ndarray):
        if y.dtype.kind in {'f', 'b'}:
            y = (y > 0.5).astype(np.int64)
        return torch.tensor(y, dtype=torch.long)
    elif torch.is_tensor(y):
        return y.long()
    else:
        raise TypeError("Expected numpy array or torch tensor")


def _make_loaders(X_train, y_train, X_val, y_val, batch_size=128, shuffle=True):
    X_train = _to_tensor(X_train)
    X_val = _to_tensor(X_val)
    y_train = _to_long(y_train)
    y_val = _to_long(y_val)

    train_ds = TensorDataset(X_train, y_train)
    val_ds = TensorDataset(X_val, y_val)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=shuffle)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    input_dim = X_train.shape[-1]
    return train_loader, val_loader, input_dim


def _compute_metrics(y_true, y_prob, threshold=0.5) -> Dict[str, float]:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    y_pred = (y_prob >= threshold).astype(int)

    out = {
        "acc": accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred, average="binary"),
    }
    try:
        out["roc_auc"] = roc_auc_score(y_true, y_prob)
    except Exception:
        out["roc_auc"] = float("nan")
    return out


def summarize_run(history: Dict[str, List[float]], best: Dict[str, Any]) -> str:
    lines = []
    lines.append("Best trial (by val_f1):")
    for k, v in best.items():
        if k in ("state_dict", "model"):
            continue
        lines.append(f"  {k}: {v}")
    lines.append("")
    if history:
        lines.append("Learning curves (last 5 epochs):")
        for key in ("train_loss", "val_loss", "train_f1", "val_f1", "val_roc_auc", "val_acc"):
            if key in history and len(history[key]) > 0:
                tail = [f"{x:.4f}" for x in history[key][-5:]]
                lines.append(f"  {key}: " + ", ".join(tail))
    return "\n".join(lines)


# --------------------------
# MLP
# --------------------------

class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_sizes: List[int], dropout: float = 0.0, bn: bool = True):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_sizes:
            layers += [nn.Linear(prev, h)]
            if bn:
                layers += [nn.BatchNorm1d(h)]
            layers += [nn.ReLU(inplace=True)]
            if dropout > 0:
                layers += [nn.Dropout(dropout)]
            prev = h
        layers += [nn.Linear(prev, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        if x.ndim > 2:
            x = x.reshape(x.size(0), -1)
        return self.net(x).squeeze(1)


@dataclass
class TrainConfig:
    lr: float = 1e-3
    weight_decay: float = 1e-4
    max_epochs: int = 50
    patience: int = 8
    pos_weight: Optional[float] = None


def _train_val_loop(model, train_loader, val_loader, cfg: TrainConfig):
    model = model.to(DEVICE)
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=(torch.tensor([cfg.pos_weight]).to(DEVICE) if cfg.pos_weight else None)
    )
    optimizer = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    history = {"train_loss": [], "val_loss": [], "train_f1": [], "val_f1": [], "val_roc_auc": [], "val_acc": []}
    best_state = None
    best_f1 = -1.0
    no_improve = 0

    for epoch in range(cfg.max_epochs):
        model.train()
        tloss = 0.0
        y_true_tr, y_prob_tr = [], []
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            logits = model(xb)
            loss = criterion(logits, yb.float())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            tloss += loss.item() * xb.size(0)
            y_prob_tr.extend(torch.sigmoid(logits).detach().cpu().numpy().tolist())
            y_true_tr.extend(yb.detach().cpu().numpy().tolist())

        tloss /= len(train_loader.dataset)
        tr_metrics = _compute_metrics(y_true_tr, y_prob_tr)

        model.eval()
        with torch.no_grad():
            vloss = 0.0
            y_true, y_prob = [], []
            for xb, yb in val_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                logits = model(xb)
                loss = criterion(logits, yb.float())
                vloss += loss.item() * xb.size(0)
                y_prob.extend(torch.sigmoid(logits).cpu().numpy().tolist())
                y_true.extend(yb.cpu().numpy().tolist())
            vloss /= len(val_loader.dataset)
            val_metrics = _compute_metrics(y_true, y_prob)

        history["train_loss"].append(tloss)
        history["val_loss"].append(vloss)
        history["train_f1"].append(tr_metrics["f1"])
        history["val_f1"].append(val_metrics["f1"])
        history["val_roc_auc"].append(val_metrics["roc_auc"])
        history["val_acc"].append(val_metrics["acc"])

        if val_metrics["f1"] > best_f1 + 1e-4:
            best_f1 = val_metrics["f1"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= cfg.patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, history, {"val_f1": best_f1, "val_roc_auc": history["val_roc_auc"][-1]}


def search_mlp(
    X_train, y_train, X_val, y_val,
    n_trials: int = 30,
    batch_size_choices=(64, 128, 256),
    hidden_choices=((64,), (128,), (256,), (128, 64), (256, 128), (256, 128, 64)),
    dropout_range=(0.0, 0.5),
    lr_range=(1e-4, 3e-3),
    wd_range=(0.0, 1e-3),
    class_pos_weight: Optional[float] = None,
    seed: int = 42,
):
    _seed_everything(seed)
    train_loader, val_loader, input_dim = _make_loaders(X_train, y_train, X_val, y_val)

    leaderboard = []
    best = None
    best_model = None
    history_of_best = None

    for t in range(n_trials):
        batch_size = random.choice(batch_size_choices)
        hidden = random.choice(hidden_choices)
        dropout = random.uniform(*dropout_range)
        lr = 10 ** random.uniform(math.log10(lr_range[0]), math.log10(lr_range[1]))
        wd = 10 ** random.uniform(-8, math.log10(wd_range[1] + 1e-8)) if wd_range[1] > 0 else 0.0
        max_epochs = random.choice([40, 60, 80])
        patience = random.choice([6, 8, 10])

        train_loader, val_loader, _ = _make_loaders(X_train, y_train, X_val, y_val, batch_size=batch_size)

        model = MLP(input_dim=input_dim, hidden_sizes=list(hidden), dropout=dropout, bn=True)
        cfg = TrainConfig(lr=lr, weight_decay=wd, max_epochs=max_epochs, patience=patience, pos_weight=class_pos_weight)

        model, history, summary = _train_val_loop(model, train_loader, val_loader, cfg)
        trial = {
            "trial": t + 1,
            "batch_size": batch_size,
            "hidden": list(hidden),
            "dropout": round(dropout, 4),
            "lr": lr,
            "weight_decay": wd,
            "max_epochs": max_epochs,
            "patience": patience,
            "val_f1": summary["val_f1"],
            "val_roc_auc": summary["val_roc_auc"],
            "final_val_loss": history["val_loss"][-1] if history["val_loss"] else None
        }
        leaderboard.append(trial)

        if best is None or trial["val_f1"] > best["val_f1"]:
            best = trial
            best_model = model
            history_of_best = history

    leaderboard = sorted(leaderboard, key=lambda d: (-d["val_f1"], -d["val_roc_auc"]))
    return best_model, best, history_of_best, leaderboard


# --------------------------
# Transformer
# --------------------------

class TimeSeriesTransformer(nn.Module):
    def __init__(self, feature_dim: int, d_model: int = 128, nhead: int = 4,
                 num_layers: int = 2, dim_feedforward: int = 256, dropout: float = 0.1):
        super().__init__()
        self.input_proj = nn.Linear(feature_dim, d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True, activation='gelu'
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1)
        )

    def forward(self, x):
        if x.ndim != 3:
            raise ValueError("Transformer expects [batch, time, features]")
        h = self.input_proj(x)
        h = self.encoder(h)
        h = self.norm(h)
        h = h.mean(dim=1)
        out = self.head(h).squeeze(1)
        return out


def _make_seq_loaders(X_train, y_train, X_val, y_val, batch_size=64, shuffle=True):
    X_train = _to_tensor(X_train)
    X_val = _to_tensor(X_val)
    y_train = _to_long(y_train)
    y_val = _to_long(y_val)
    train_ds = TensorDataset(X_train, y_train)
    val_ds = TensorDataset(X_val, y_val)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=shuffle)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    feature_dim = X_train.shape[-1]
    return train_loader, val_loader, feature_dim


def search_transformer(
    X_train, y_train, X_val, y_val,
    n_trials: int = 30,
    batch_size_choices=(32, 64, 128),
    d_model_choices=(64, 96, 128, 192),
    nhead_choices=(2, 4, 8),
    num_layers_choices=(1, 2, 3, 4),
    ffn_multiplier_choices=(2, 4),
    dropout_range=(0.0, 0.4),
    lr_range=(5e-5, 3e-3),
    wd_range=(0.0, 1e-3),
    class_pos_weight: Optional[float] = None,
    seed: int = 42,
):
    _seed_everything(seed)
    train_loader, val_loader, feature_dim = _make_seq_loaders(X_train, y_train, X_val, y_val)

    leaderboard = []
    best = None
    best_model = None
    history_of_best = None

    for t in range(n_trials):
        batch_size = random.choice(batch_size_choices)
        d_model = random.choice(d_model_choices)
        nhead = random.choice([h for h in nhead_choices if d_model % h == 0])
        num_layers = random.choice(num_layers_choices)
        ffn_mul = random.choice(ffn_multiplier_choices)
        dim_ff = d_model * ffn_mul
        dropout = random.uniform(*dropout_range)
        lr = 10 ** random.uniform(math.log10(lr_range[0]), math.log10(lr_range[1]))
        wd = 10 ** random.uniform(-8, math.log10(wd_range[1] + 1e-8)) if wd_range[1] > 0 else 0.0
        max_epochs = random.choice([40, 60, 80])
        patience = random.choice([6, 8, 10])

        train_loader, val_loader, _ = _make_seq_loaders(X_train, y_train, X_val, y_val, batch_size=batch_size)

        model = TimeSeriesTransformer(feature_dim, d_model=d_model, nhead=nhead,
                                      num_layers=num_layers, dim_feedforward=dim_ff, dropout=dropout)
        cfg = TrainConfig(lr=lr, weight_decay=wd, max_epochs=max_epochs,
                          patience=patience, pos_weight=class_pos_weight)

        model, history, summary = _train_val_loop(model, train_loader, val_loader, cfg)
        trial = {
            "trial": t + 1,
            "batch_size": batch_size,
            "d_model": d_model,
            "nhead": nhead,
            "num_layers": num_layers,
            "dim_feedforward": dim_ff,
            "dropout": round(dropout, 4),
            "lr": lr,
            "weight_decay": wd,
            "max_epochs": max_epochs,
            "patience": patience,
            "val_f1": summary["val_f1"],
            "val_roc_auc": summary["val_roc_auc"],
            "final_val_loss": history["val_loss"][-1] if history["val_loss"] else None
        }
        leaderboard.append(trial)

        if best is None or trial["val_f1"] > best["val_f1"]:
            best = trial
            best_model = model
            history_of_best = history

    leaderboard = sorted(leaderboard, key=lambda d: (-d["val_f1"], -d["val_roc_auc"]))
    return best_model, best, history_of_best, leaderboard

# =========================
# Time-Pooled MLP (3D input)
# =========================

class TimeStepMLPToken(nn.Module):
    """
    Apply an MLP (shared over time) to each time step's features and output a logit per time step.
    Input:  x [B, T, F]
    Output: logits [B, T]
    """
    def __init__(self, feature_dim: int, hidden_sizes: List[int], dropout: float=0.2, bn: bool=True):
        super().__init__()
        layers = []
        prev = feature_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(prev, h))
            if bn:
                layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU(inplace=True))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, 1))  # per-token logit
        self.net = nn.Sequential(*layers)

    def forward(self, x):  # x: [B, T, F]
        if x.ndim != 3:
            raise ValueError("TimeStepMLPToken expects [batch, time, features]")
        B, T, F = x.shape
        h = x.reshape(B*T, F)
        logits = self.net(h).squeeze(1)   # [B*T]
        return logits.view(B, T)          # [B, T]


class TimeSeriesTransformerToken(nn.Module):
    """
    Transformer encoder that outputs a logit per time step (token classification).
    Input:  x [B, T, F]
    Output: logits [B, T]
    """
    def __init__(self, feature_dim: int, d_model: int=128, nhead: int=4,
                 num_layers: int=2, dim_feedforward: int=256, dropout: float=0.1):
        super().__init__()
        self.input_proj = nn.Linear(feature_dim, d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True, activation='gelu'
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, 1)  # per-token logit

    def forward(self, x):  # x: [B, T, F]
        if x.ndim != 3:
            raise ValueError("TimeSeriesTransformerToken expects [batch, time, features]")
        h = self.input_proj(x)             # [B, T, d_model]
        h = self.encoder(h)                # [B, T, d_model]
        h = self.norm(h)                   # [B, T, d_model]
        logits = self.head(h).squeeze(-1)  # [B, T]
        return logits


# --------------------------
# Loaders & metrics for token labels y [B, T]
# --------------------------

def _make_seq_token_loaders(X_train, y_train, X_val, y_val, batch_size=64, shuffle=True):
    X_train = _to_tensor(X_train)  # [N, T, F]
    X_val   = _to_tensor(X_val)
    # y are per-time-step labels -> expect [N, T] (ints 0/1)
    if isinstance(y_train, np.ndarray):
        y_train = torch.tensor(y_train, dtype=torch.long)
        y_val   = torch.tensor(y_val, dtype=torch.long)
    elif torch.is_tensor(y_train):
        y_train = y_train.long()
        y_val   = y_val.long()
    else:
        raise TypeError("y must be numpy array or torch tensor")

    assert X_train.ndim == 3 and X_val.ndim == 3, "X must be [N, T, F]"
    assert y_train.ndim == 2 and y_val.ndim == 2, "y must be [N, T] with 0/1 ints"
    assert X_train.shape[0] == y_train.shape[0] and X_train.shape[1] == y_train.shape[1], "X and y time dims must match"
    assert X_val.shape[0] == y_val.shape[0] and X_val.shape[1] == y_val.shape[1], "X_val and y_val time dims must match"

    train_ds = TensorDataset(X_train, y_train)
    val_ds   = TensorDataset(X_val,   y_val)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=shuffle, drop_last=False)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,    drop_last=False)
    feature_dim  = X_train.shape[-1]
    return train_loader, val_loader, feature_dim


def _compute_metrics_token(y_true_2d, y_prob_2d, threshold=0.5) -> Dict[str, float]:
    """
    y_true_2d: [N, T] ints {0,1}
    y_prob_2d: [N, T] floats in [0,1]
    """
    yt = np.asarray(y_true_2d).reshape(-1).astype(int)
    yp = np.asarray(y_prob_2d).reshape(-1).astype(float)
    yhat = (yp >= threshold).astype(int)
    out = {
        "acc": accuracy_score(yt, yhat),
        "f1":  f1_score(yt, yhat, average="binary"),
    }
    try:
        out["roc_auc"] = roc_auc_score(yt, yp)
    except Exception:
        out["roc_auc"] = float("nan")
    return out


def _train_val_loop_token(model, train_loader, val_loader, cfg: TrainConfig):
    """
    Train loop for token classification (per-time-step). Uses BCEWithLogits over [B, T].
    Early stopping on val F1 (flattened over all tokens).
    """
    model = model.to(DEVICE)
    pos_w = None
    if cfg.pos_weight is not None:
        pos_w = torch.tensor([cfg.pos_weight], dtype=torch.float32, device=DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    optimizer = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    history = {"train_loss":[], "val_loss":[], "train_f1":[], "val_f1":[], "val_roc_auc":[], "val_acc":[]}
    best_state = None
    best_f1 = -1.0
    no_improve = 0

    for epoch in range(cfg.max_epochs):
        # ---- Train
        model.train()
        tloss = 0.0
        tr_probs, tr_true = [], []
        for xb, yb in train_loader:
            xb = xb.to(DEVICE)           # [B, T, F]
            yb = yb.to(DEVICE)           # [B, T] ints
            logits = model(xb)           # [B, T]
            loss = criterion(logits, yb.float())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            tloss += loss.item() * xb.size(0)
            tr_probs.append(torch.sigmoid(logits).detach().cpu().numpy())
            tr_true.append(yb.detach().cpu().numpy())

        tloss /= len(train_loader.dataset)
        tr_probs = np.concatenate(tr_probs, axis=0)
        tr_true  = np.concatenate(tr_true,  axis=0)
        tr_metrics = _compute_metrics_token(tr_true, tr_probs)

        # ---- Validate
        model.eval()
        vloss = 0.0
        va_probs, va_true = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(DEVICE)
                yb = yb.to(DEVICE)
                logits = model(xb)
                loss = criterion(logits, yb.float())
                vloss += loss.item() * xb.size(0)
                va_probs.append(torch.sigmoid(logits).cpu().numpy())
                va_true.append(yb.cpu().numpy())
        vloss /= len(val_loader.dataset)
        va_probs = np.concatenate(va_probs, axis=0)
        va_true  = np.concatenate(va_true,  axis=0)
        val_metrics = _compute_metrics_token(va_true, va_probs)

        # Track
        history["train_loss"].append(tloss)
        history["val_loss"].append(vloss)
        history["train_f1"].append(tr_metrics["f1"])
        history["val_f1"].append(val_metrics["f1"])
        history["val_roc_auc"].append(val_metrics["roc_auc"])
        history["val_acc"].append(val_metrics["acc"])

        # Early stopping
        if val_metrics["f1"] > best_f1 + 1e-4:
            best_f1 = val_metrics["f1"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= cfg.patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, history, {"val_f1": best_f1, "val_roc_auc": history["val_roc_auc"][-1] if history["val_roc_auc"] else float('nan')}


# --------------------------
# Searches for token models
# --------------------------

def search_timestep_mlp_token(
    X_train, y_train, X_val, y_val,
    n_trials: int = 30,
    batch_size_choices=(32, 64, 128),
    hidden_choices=((128,), (256,), (256,128), (256,128,64)),
    dropout_range=(0.0, 0.5),
    lr_range=(1e-4, 3e-3),
    wd_range=(0.0, 1e-3),
    class_pos_weight: Optional[float] = None,
    seed: int = 42,
):
    _seed_everything(seed)
    train_loader, val_loader, feature_dim = _make_seq_token_loaders(
        X_train, y_train, X_val, y_val, batch_size=random.choice(batch_size_choices)
    )

    leaderboard = []
    best = None
    best_model = None
    history_of_best = None

    for t in range(n_trials):
        batch_size = random.choice(batch_size_choices)
        hidden = list(random.choice(hidden_choices))
        dropout = random.uniform(*dropout_range)
        lr = 10 ** random.uniform(math.log10(lr_range[0]), math.log10(lr_range[1]))
        wd = 0.0 if wd_range[1] == 0 else 10 ** random.uniform(math.log10(max(wd_range[0],1e-8)), math.log10(max(wd_range[1],1e-8)))
        max_epochs = random.choice([40, 60, 80])
        patience = random.choice([6, 8, 10])

        train_loader, val_loader, _ = _make_seq_token_loaders(
            X_train, y_train, X_val, y_val, batch_size=batch_size
        )

        model = TimeStepMLPToken(feature_dim=feature_dim, hidden_sizes=hidden, dropout=dropout, bn=True)
        cfg = TrainConfig(lr=lr, weight_decay=wd, max_epochs=max_epochs, patience=patience, pos_weight=class_pos_weight)

        model, history, summary = _train_val_loop_token(model, train_loader, val_loader, cfg)
        trial = {
            "trial": t+1,
            "batch_size": batch_size,
            "hidden": hidden,
            "dropout": round(dropout,4),
            "lr": lr,
            "weight_decay": wd,
            "max_epochs": max_epochs,
            "patience": patience,
            "val_f1": summary["val_f1"],
            "val_roc_auc": summary["val_roc_auc"],
            "final_val_loss": history["val_loss"][-1] if history["val_loss"] else None
        }
        leaderboard.append(trial)

        if best is None or trial["val_f1"] > best["val_f1"]:
            best = trial
            best_model = model
            history_of_best = history

    leaderboard = sorted(leaderboard, key=lambda d: (-d["val_f1"], - (d["val_roc_auc"] if d["val_roc_auc"]==d["val_roc_auc"] else -1)))
    return best_model, best, history_of_best, leaderboard


def search_transformer_token(
    X_train, y_train, X_val, y_val,
    n_trials: int = 30,
    batch_size_choices=(16, 32, 64),
    d_model_choices=(64, 96, 128, 192),
    nhead_choices=(2, 4, 8),
    num_layers_choices=(1, 2, 3, 4),
    ffn_multiplier_choices=(2, 4),
    dropout_range=(0.0, 0.4),
    lr_range=(5e-5, 3e-3),
    wd_range=(0.0, 1e-3),
    class_pos_weight: Optional[float] = None,
    seed: int = 42,
):
    _seed_everything(seed)
    train_loader, val_loader, feature_dim = _make_seq_token_loaders(
        X_train, y_train, X_val, y_val, batch_size=random.choice(batch_size_choices)
    )

    leaderboard = []
    best = None
    best_model = None
    history_of_best = None

    for t in range(n_trials):
        batch_size = random.choice(batch_size_choices)
        d_model = random.choice(d_model_choices)
        nhead = random.choice([h for h in nhead_choices if d_model % h == 0])
        num_layers = random.choice(num_layers_choices)
        ffn_mul = random.choice(ffn_multiplier_choices)
        dim_ff = d_model * ffn_mul
        dropout = random.uniform(*dropout_range)
        lr = 10 ** random.uniform(math.log10(lr_range[0]), math.log10(lr_range[1]))
        wd = 0.0 if wd_range[1] == 0 else 10 ** random.uniform(math.log10(max(wd_range[0],1e-8)), math.log10(max(wd_range[1],1e-8)))
        max_epochs = random.choice([40, 60, 80])
        patience = random.choice([6, 8, 10])

        train_loader, val_loader, _ = _make_seq_token_loaders(
            X_train, y_train, X_val, y_val, batch_size=batch_size
        )

        model = TimeSeriesTransformerToken(
            feature_dim=feature_dim, d_model=d_model, nhead=nhead,
            num_layers=num_layers, dim_feedforward=dim_ff, dropout=dropout
        )
        cfg = TrainConfig(lr=lr, weight_decay=wd, max_epochs=max_epochs, patience=patience, pos_weight=class_pos_weight)

        model, history, summary = _train_val_loop_token(model, train_loader, val_loader, cfg)
        trial = {
            "trial": t+1,
            "batch_size": batch_size,
            "d_model": d_model,
            "nhead": nhead,
            "num_layers": num_layers,
            "dim_feedforward": dim_ff,
            "dropout": round(dropout,4),
            "lr": lr,
            "weight_decay": wd,
            "max_epochs": max_epochs,
            "patience": patience,
            "val_f1": summary["val_f1"],
            "val_roc_auc": summary["val_roc_auc"],
            "final_val_loss": history["val_loss"][-1] if history["val_loss"] else None
        }
        leaderboard.append(trial)

        if best is None or trial["val_f1"] > best["val_f1"]:
            best = trial
            best_model = model
            history_of_best = history

    leaderboard = sorted(leaderboard, key=lambda d: (-d["val_f1"], - (d["val_roc_auc"] if d["val_roc_auc"]==d["val_roc_auc"] else -1)))
    return best_model, best, history_of_best, leaderboard

# ---- Transformers ------

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.bce = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, logits, targets):
        bce_loss = self.bce(logits, targets)
        probs = torch.sigmoid(logits)
        pt = torch.where(targets == 1, probs, 1 - probs)
        loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        if self.reduction == 'mean':
            return loss.mean()
        return loss.sum()
    
    
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :]
    
    
class REM_Transformer(nn.Module):
    def __init__(self, input_dim=2, d_model=64, nhead=4, num_layers=2,
                 dim_feedforward=128, dropout=0.2, max_len=5000):
        super().__init__()
        self.input_fc = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Linear(d_model, 1)

    def forward(self, x, src_key_padding_mask=None):
        x = self.input_fc(x)
        x = self.pos_encoder(x)
        x = self.transformer(x, src_key_padding_mask=src_key_padding_mask)
        last_token = x[:, -1, :]
        return self.classifier(last_token).squeeze(1)
    

def create_transformer_sequences_dynamic(X, y, max_window_size):
    X_seq, y_seq, masks = [], [], []
    n_trials, n_bins, n_features = X.shape
    for trial in range(n_trials):
        for b in range(1, n_bins):
            start = max(0, b - max_window_size)
            x_window = X[trial, start:b]
            pad_len = max_window_size - x_window.shape[0]
            pad = np.zeros((pad_len, n_features))
            x_padded = np.vstack((pad, x_window))
            mask = np.zeros(max_window_size, dtype=bool)
            mask[:pad_len] = True
            X_seq.append(x_padded)
            masks.append(mask)
            y_seq.append(y[trial, b])
    return np.array(X_seq), np.array(y_seq), np.array(masks)

# ---- Tensors ----
def to_tensors(X, y, m):
    return (torch.tensor(X, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32),
            torch.tensor(m, dtype=torch.bool))


# ---- Weighted sampler over sequences (handles imbalance) ----
def make_loader(X, y, m, batch_size, balance=True, shuffle=False):
    ds = TensorDataset(X, y, m)
    if balance:
        counts = np.bincount(y.cpu().numpy().astype(int))
        w = 1.0 / np.maximum(counts, 1)
        sw = w[y.cpu().numpy().astype(int)]
        sampler = WeightedRandomSampler(sw, len(sw), replacement=True)
        return DataLoader(ds, batch_size=batch_size, sampler=sampler, drop_last=False)
    else:
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=False)

# ---- Train/eval for one config (early stopping on val F1 with threshold sweep) ----
def train_eval_config(X_train, y_train, m_train, X_val, y_val, m_val, X_seq, cfg, verbose=False, device=torch.device("cpu")):
    """
    cfg dict keys:
      d_model, nhead, num_layers, ffn_mul, dropout, lr, batch_size, max_epochs, patience
    Returns best_val_f1, best_thresh, best_state_dict, history (dict)
    """
    batch_size = cfg["batch_size"]
    train_loader = make_loader(X_train, y_train, m_train, batch_size=batch_size, balance=True)
    val_loader   = make_loader(X_val,   y_val,   m_val,   batch_size=batch_size, balance=False)

    model = REM_Transformer(
        input_dim=X_seq.shape[-1],
        d_model=cfg["d_model"],
        nhead=cfg["nhead"],
        num_layers=cfg["num_layers"],
        dim_feedforward=cfg["d_model"] * cfg["ffn_mul"],
        dropout=cfg["dropout"]
    ).to(device)

    criterion = FocalLoss(alpha=0.5, gamma=2.0)
    optimizer = optim.Adam(model.parameters(), lr=cfg["lr"])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=3)

    best_f1, best_thresh, wait = 0.0, 0.5, 0
    best_state = None
    history = {"val_f1": [], "val_loss": []}

    for epoch in range(1, cfg["max_epochs"]+1):
        # --- Train ---
        model.train()
        train_loss = 0.0
        for xb, yb, mb in train_loader:
            xb, yb, mb = xb.to(device), yb.to(device), mb.to(device)
            optimizer.zero_grad()
            logits = model(xb, src_key_padding_mask=mb)    # [B]
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # --- Validate ---
        model.eval()
        val_loss = 0.0
        probs_all, labels_all = [], []
        with torch.no_grad():
            for xb, yb, mb in val_loader:
                xb, yb, mb = xb.to(device), yb.to(device), mb.to(device)
                logits = model(xb, src_key_padding_mask=mb)
                loss = criterion(logits, yb)
                val_loss += loss.item()
                probs_all.append(torch.sigmoid(logits).detach().cpu().numpy())
                labels_all.append(yb.detach().cpu().numpy())
        probs = np.concatenate(probs_all)
        labels = np.concatenate(labels_all)

        # threshold sweep for F1
        prec, rec, ths = precision_recall_curve(labels, probs)
        f1_scores = 2 * (prec * rec) / (prec + rec + 1e-8)
        if f1_scores.size > 0:
            idx = int(np.argmax(f1_scores))
            cur_f1 = float(f1_scores[idx])
            cur_thresh = float(ths[max(idx, 0)]) if ths.size > 0 else 0.5
        else:
            cur_f1, cur_thresh = 0.0, 0.5

        history["val_f1"].append(cur_f1)
        history["val_loss"].append(val_loss)

        if verbose and (epoch == 1 or epoch % 5 == 0):
            print(f"Epoch {epoch:03d} | val_loss={val_loss:.4f} | val_f1={cur_f1:.4f} | best={best_f1:.4f}")

        scheduler.step(val_loss)

        # early stopping on F1
        if cur_f1 > best_f1 + 1e-4:
            best_f1, best_thresh, wait = cur_f1, cur_thresh, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= cfg["patience"]:
                break

    return best_f1, best_thresh, best_state, history, model.state_dict()

# ---- Random search space ----
def sample_cfg():
    d_model_choices = [64, 96, 128, 192]
    d_model = random.choice(d_model_choices)
    nhead_choices = [h for h in [2, 4, 8] if d_model % h == 0]
    cfg = {
        "d_model": d_model,
        "nhead": random.choice(nhead_choices),
        "num_layers": random.choice([1, 2, 3, 4]),
        "ffn_mul": random.choice([2, 4]),
        "dropout": random.uniform(0.0, 0.3),
        "lr": 10 ** random.uniform(math.log10(5e-5), math.log10(3e-3)),
        "batch_size": random.choice([32, 64, 128]),
        "max_epochs": random.choice([40, 60, 80]),
        "patience": random.choice([6, 8, 10]),
    }
    return cfg