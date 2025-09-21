# src/visualize.py
"""
  Common training/evaluation visualizations for fast diagnostics and report figures.

  - plot_training_curves(history, out_path): loss & metric curves across epochs
  - plot_confusion_matrix(cm, classes, out_path): heatmap of test confusion
  - plot_roc_curves(y_true, logits, classes, out_path): one-vs-rest ROC curves + macro AUC
  - plot_per_class_report(y_true, logits, classes, out_path): per-class Precision/Recall/F1 bars
  - plot_prob_histogram(y_true, logits, out_path): confidence hist (max-prob) for correct vs wrong

  - history: list of dicts per epoch with keys:
      {'tr_loss', 'va_loss', 'va_acc', 'va_f1', 'va_auc'}  (you’ll record these)
  - y_true: 1D np.array of int labels (N,)
  - logits: 2D np.array of raw scores (N, C)
  - cm:     2D np.array confusion matrix (C, C)
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_fscore_support

def _ensure_parent(out_path):
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

def _softmax_np(x: np.ndarray) -> np.ndarray:
    x = x - x.max(axis=1, keepdims=True)
    ex = np.exp(x)
    return ex / np.clip(ex.sum(axis=1, keepdims=True), 1e-12, None)

def plot_training_curves(history, out_path):
    """
    Plots train/val loss and val metrics (Acc/F1/AUC) vs. epoch.
    To eee overfitting/underfitting and metric trends at a glance.
     
    """
    _ensure_parent(out_path)
    epochs = np.arange(1, len(history) + 1)
    tr_loss = [h['tr_loss'] for h in history]
    va_loss = [h['va_loss'] for h in history]
    va_acc  = [h['va_acc']  for h in history]
    va_f1   = [h['va_f1']   for h in history]
    va_auc  = [h['va_auc']  for h in history]

    # Loss
    plt.figure(figsize=(6,4))
    plt.plot(epochs, tr_loss, label="train loss")
    plt.plot(epochs, va_loss, label="val loss")
    plt.xlabel("epoch"); plt.ylabel("loss"); plt.title("Loss vs. epoch"); plt.legend(); plt.tight_layout()
    plt.savefig(str(Path(out_path).with_suffix("").as_posix() + "_loss.png"), dpi=200)
    plt.close()

    # Metrics
    plt.figure(figsize=(6,4))
    plt.plot(epochs, va_acc, label="val acc")
    plt.plot(epochs, va_f1,  label="val macro-F1")
    plt.plot(epochs, va_auc, label="val macro-AUC")
    plt.xlabel("epoch"); plt.ylabel("score"); plt.title("Val metrics vs. epoch"); plt.legend(); plt.tight_layout()
    plt.savefig(str(Path(out_path).with_suffix("").as_posix() + "_metrics.png"), dpi=200)
    plt.close()

def plot_confusion_matrix(cm: np.ndarray, classes, out_path):
    """
    Heatmap of confusion counts with class tick labels.
    To identify systematic confusions between classes.
    """
    _ensure_parent(out_path)
    plt.figure(figsize=(6,5))
    im = plt.imshow(cm, interpolation='nearest')
    plt.title("Confusion Matrix")
    plt.xticks(range(len(classes)), classes, rotation=45, ha="right")
    plt.yticks(range(len(classes)), classes)
    thresh = cm.max() / 2.0 if cm.size else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i, j], ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black", fontsize=9)
    plt.ylabel("True"); plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

def plot_roc_curves(y_true: np.ndarray, logits: np.ndarray, classes, out_path):
    """
    One-vs-rest ROC curves and macro AUC.
    To show discrimination quality per class and overall.
    """
    _ensure_parent(out_path)
    probs = _softmax_np(logits)
    n_classes = probs.shape[1]
    plt.figure(figsize=(6,5))
    aucs = []
    for c in range(n_classes):
        y_bin = (y_true == c).astype(int)
        fpr, tpr, _ = roc_curve(y_bin, probs[:, c])
        aucs.append(auc(fpr, tpr))
        plt.plot(fpr, tpr, label=f"{classes[c]} (AUC={aucs[-1]:.3f})")
    plt.plot([0,1],[0,1], linestyle="--")
    plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title("One-vs-Rest ROC")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

def plot_per_class_report(y_true: np.ndarray, logits: np.ndarray, classes, out_path):
    """
    Bar chart of per-class P/R/F1 using argmax predictions.
    To see which classes have low precision/recall/F1.
    """
    _ensure_parent(out_path)
    preds = _softmax_np(logits).argmax(1)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, preds, labels=range(len(classes)))
    x = np.arange(len(classes))
    plt.figure(figsize=(7,4))
    plt.bar(x - 0.25, prec, width=0.25, label="Precision")
    plt.bar(x,         rec,  width=0.25, label="Recall")
    plt.bar(x + 0.25,  f1,   width=0.25, label="F1")
    plt.xticks(x, classes, rotation=45, ha="right")
    plt.ylim(0,1)
    plt.title("Per-class Precision/Recall/F1")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

def plot_prob_histogram(y_true: np.ndarray, logits: np.ndarray, out_path):
    """
    Histogram of max predicted probability for correct vs incorrect samples.
    To inspect calibration-ish behavior: are correct predictions more confident?
    """
    _ensure_parent(out_path)
    probs = _softmax_np(logits)
    pred = probs.argmax(1)
    maxp = probs.max(1)
    correct = maxp[pred == y_true]
    wrong   = maxp[pred != y_true]

    plt.figure(figsize=(6,4))
    plt.hist(correct, bins=20, alpha=0.6, label="correct")
    plt.hist(wrong,   bins=20, alpha=0.6, label="wrong")
    plt.xlabel("max predicted probability")
    plt.ylabel("count")
    plt.title("Prediction Confidence (Correct vs Wrong)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
