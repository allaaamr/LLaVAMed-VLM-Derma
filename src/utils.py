
from __future__ import annotations
import numpy as np, torch
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix

def set_seed(seed: int = 42):
    import random
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def choose_device() -> torch.device:
    if torch.cuda.is_available(): return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available(): return torch.device("mps")
    return torch.device("cpu")

def softmax_np(x: np.ndarray) -> np.ndarray:
    x = x - x.max(axis=1, keepdims=True)
    ex = np.exp(x)
    return ex / np.clip(ex.sum(axis=1, keepdims=True), 1e-12, None)

def compute_metrics(y_true: np.ndarray, logits: np.ndarray) -> dict:
    probs = softmax_np(logits); preds = probs.argmax(1)
    return {
        "accuracy": float(accuracy_score(y_true, preds)),
        "f1_macro": float(f1_score(y_true, preds, average="macro")),
        "auc_ovr": float(roc_auc_score(y_true, probs, multi_class="ovr", average="macro")),
        "confusion": confusion_matrix(y_true, preds).tolist(),
    }

def run_epoch(model, loader, device, criterion, optimizer=None):
    """
    WHY: One routine for train or eval depending on optimizer presence.
    RETURNS: avg_loss, logits(np.ndarray), labels(np.ndarray)
    """
    is_train = optimizer is not None
    model.train(is_train)
    losses, logits_all, labels_all = [], [], []
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        with torch.set_grad_enabled(is_train):
            logits = model(xb)
            loss = criterion(logits, yb)
            if is_train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward(); optimizer.step()
        losses.append(loss.item())
        logits_all.append(logits.detach().cpu().numpy())
        labels_all.append(yb.detach().cpu().numpy())
    avg = float(np.mean(losses)) if losses else 0.0
    L = np.concatenate(logits_all, 0) if logits_all else np.zeros((0, getattr(model, "num_classes", 7)))
    y = np.concatenate(labels_all, 0) if labels_all else np.zeros((0,), dtype=np.int64)
    return avg, L, y
