# src/tasks/train_baseline.py
import yaml, numpy as np, torch, torch.nn as nn
from src.data import make_loaders
from src.utils import set_seed, choose_device, run_epoch, compute_metrics
from src.models.baseline import build_timm_model
from pathlib import Path
from src.visualize import  plot_training_curves, plot_confusion_matrix, plot_roc_curves, plot_per_class_report, plot_prob_histogram


"""
  Train a 7-class baseline (CNN/ViT) using the same csv splits 
"""

def class_weights_from_train(train_labels: np.ndarray, n_classes: int, cap: float = 3.0) -> np.ndarray:
    """
    Address class imbalance so minority classes contribute more to the loss.
    Balanced weights = N / (K * count_c); clipped by 'cap' to avoid huge gradients.

    Parameters:
      - train_labels: integer class indices from the TRAIN split
      - n_classes   : number of classes (7)
      - cap         : maximum allowed weight per class (from config)
    Returns:
      - weights (np.ndarray shape [n_classes])
    """
    counts = np.bincount(train_labels, minlength=n_classes).astype(float) # per-class counts
    w = (len(train_labels) / (n_classes * np.clip(counts, 1, None)))  # "balanced" formula
    return np.minimum(w, cap) # cap for stability

def main():
    cfg = yaml.safe_load(open("configs.yaml"))
    set_seed(cfg["seed"]); device = choose_device()
    fig_dir = Path("figures")
    history = []  # collect stats per epoch for curves

    # data/loaders (train has augmentation, val/test are deterministic)
    ds, dl, classes = make_loaders(cfg["baseline"]["batch_size"], cfg)
    n = len(classes)

    # ---- model: pick any timm architecture by name from config ----
    #  "resnet18", "resnet50", "vit_base_patch16_224", etc.
    model = build_timm_model(cfg["baseline"]["model"], num_classes=n).to(device)

    # weighted CE loss (optional cap from config)
    ytr = np.array(ds["train"].labels)
    cap = float(cfg.get("class_weighting", {}).get("cap", 3.0))
    use_w = bool(cfg.get("class_weighting", {}).get("enabled", True))
    if use_w:
        w = class_weights_from_train(ytr, n_classes=n, cap=cap)
        w_t = torch.tensor(w, dtype=torch.float32, device=device)
        criterion = nn.CrossEntropyLoss(weight=w_t)
        print(f"[class-weights] {np.round(w,3).tolist()}")
    else:
        criterion = nn.CrossEntropyLoss()
        print("[class-weights] disabled")

    optim = torch.optim.AdamW(model.parameters(), lr=float(cfg["baseline"]["lr"]), weight_decay=1e-4)

    # train with in-memory best state (no checkpoint file)
    best_f1, best_state = -1.0, None
    for ep in range(1, int(cfg["baseline"]["epochs"]) + 1):
        # 1) one training epoch
        tr_loss, _, _ = run_epoch(model, dl["train"], device, criterion, optimizer=optim)
        # 2) validation (no optimizer) to pick the best model state
        va_loss, Lval, yval = run_epoch(model, dl["val"], device, criterion, optimizer=None)
        mval = compute_metrics(yval, Lval)
        print(f"[{ep:03d}] train_loss={tr_loss:.4f} | val_loss={va_loss:.4f} "
              f"| val_acc={mval['accuracy']:.3f} | val_f1={mval['f1_macro']:.3f} | val_auc={mval['auc_ovr']:.3f}")
        
        history.append({
            "tr_loss": tr_loss,
            "va_loss": va_loss,
            "va_acc":  mval["accuracy"],
            "va_f1":   mval["f1_macro"],
            "va_auc":  mval["auc_ovr"],
        })

        if mval["f1_macro"] > best_f1:
            best_f1 = mval["f1_macro"]
            # keep a CPU copy of weights (minimal, no file I/O)
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    # load best weights back into the model
    if best_state is not None:
        model.load_state_dict(best_state, strict=True)
        model.to(device)

    # test
    te_loss, Ltest, ytest = run_epoch(model, dl["test"], device, criterion, optimizer=None)
    mtest = compute_metrics(ytest, Ltest)
    print("\n[test metrics]")
    print(mtest)

    # visualizations 
    plot_training_curves(history, fig_dir / "baseline_curves.png")
    plot_confusion_matrix(np.array(mtest["confusion"]), classes, fig_dir / "baseline_confusion.png")
    plot_roc_curves(ytest, Ltest, classes, fig_dir / "baseline_roc.png")
    plot_per_class_report(ytest, Ltest, classes, fig_dir / "baseline_prf.png")
    plot_prob_histogram(ytest, Ltest, fig_dir / "baseline_confidence.png")
    print(f"[figures] saved to {fig_dir.resolve()}")


if __name__ == "__main__":
    main()
