"""Training loop for the shared network."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch

from armd.config import Config
from armd.model import SharedNet, joint_loss


@dataclass
class TrainedModel:
    model: SharedNet
    history: dict = field(default_factory=lambda: {"train": [], "val": []})
    best_val: float = float("inf")
    epochs_run: int = 0
    seconds: float = 0.0


def pick_device(requested: Optional[str] = None) -> str:
    if requested:
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def train(
    model: SharedNet,
    tensors: dict,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    cfg: Config,
    device: str,
    verbose: bool = True,
) -> TrainedModel:
    import time

    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    train_t = torch.tensor(np.asarray(train_idx), device=device, dtype=torch.long)
    val_t = torch.tensor(np.asarray(val_idx), device=device, dtype=torch.long)
    w = cfg.causal_loss_weight

    def batch_loss(idx):
        out = model(tensors["base"][idx], tensors["expo"][idx], tensors["sev"][idx])
        return joint_loss(
            out, tensors["ys"][idx], tensors["ms"][idx],
            tensors["a"][idx], tensors["yc"][idx], tensors["mc"][idx], w,
        )

    def evaluate(idx):
        if not len(idx):
            return float("nan")
        model.eval()
        with torch.no_grad():
            return float(batch_loss(idx)[0])

    result = TrainedModel(model=model)
    best_state, wait = None, 0
    started = time.time()

    for epoch in range(cfg.max_epochs):
        model.train()
        perm = train_t[torch.randperm(len(train_t), device=device)]
        running, n_batches = 0.0, 0
        for start in range(0, len(perm), cfg.batch_size):
            batch = perm[start:start + cfg.batch_size]
            if len(batch) < 2:
                continue
            opt.zero_grad()
            loss = batch_loss(batch)[0]
            loss.backward()
            opt.step()
            running += float(loss.detach())
            n_batches += 1

        tr = running / max(n_batches, 1)
        vl = evaluate(val_t)
        result.history["train"].append(tr)
        result.history["val"].append(vl)
        result.epochs_run = epoch + 1

        improved = np.isfinite(vl) and vl < result.best_val - 1e-4
        if improved:
            result.best_val = vl
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
        if verbose and (epoch % 5 == 0 or improved):
            print(f"  epoch {epoch:3d}  train {tr:.4f}  val {vl:.4f}")
        if wait >= cfg.patience:
            if verbose:
                print(f"  early stop at epoch {epoch}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    result.seconds = time.time() - started
    return result


def predict(model: SharedNet, tensors: dict, idx: np.ndarray, device: str,
            batch_size: int = 8192) -> dict:
    """Batched inference, so scoring a large test period cannot exhaust device memory."""
    model.eval()
    succ, prop, m0, m1 = [], [], [], []
    with torch.no_grad():
        for start in range(0, len(idx), batch_size):
            chunk = torch.tensor(np.asarray(idx[start:start + batch_size]),
                                 device=device, dtype=torch.long)
            s, p, y0, y1 = model(tensors["base"][chunk], tensors["expo"][chunk], tensors["sev"][chunk])
            succ.append(torch.sigmoid(s).cpu().numpy())
            prop.append(torch.sigmoid(p).cpu().numpy())
            m0.append(torch.sigmoid(y0).cpu().numpy())
            m1.append(torch.sigmoid(y1).cpu().numpy())
    empty = np.zeros((0,), dtype=np.float32)
    return {
        "susceptibility": np.concatenate(succ) if succ else np.zeros((0, 0), np.float32),
        "propensity": np.concatenate(prop) if prop else empty,
        "m0": np.concatenate(m0) if m0 else empty,
        "m1": np.concatenate(m1) if m1 else empty,
    }
