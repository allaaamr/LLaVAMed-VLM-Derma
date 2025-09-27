import argparse
import json
from pathlib import Path
from typing import List

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import yaml

from src.vqa_utils import (
    build_prompt,
    parse_to_label,
    load_split_items,
    open_rgb_image,
    load_vlm_model,
)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs.yaml")
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--save_csv", action="store_true", help="Write per-sample preds CSV")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    classes: List[str] = cfg["data"]["classes"]                   
    vqa_dir = Path(cfg["paths"].get("vqa", "data/processed/vqa"))
    model_id = cfg["vlm"]["model_id"]
    prompt_tmpl = cfg["vlm"]["prompt_template"]
    max_new = int(cfg["vlm"].get("max_new_tokens", 8))
    temperature = float(cfg["vlm"].get("temperature", 0.0))
    device_map = cfg["vlm"].get("device_map", "auto")
    dtype = cfg["vlm"].get("dtype", "auto")

    # ------------------ Load items --------------------
    jsonl_path = vqa_dir / f"{args.split}.jsonl"
    items = load_split_items(jsonl_path)
    if not items:
        print(f"[error] No items loaded from {jsonl_path}")
        return

    # ------------------ Load model --------------------
    processor, model = load_vlm_model(model_id, device_map=device_map, dtype=dtype)
    model.eval()

    # ------------------ Evaluate ----------------------
    options = ",".join(classes)                          # e.g. "akiec,bcc,bkl,df,mel,nv,vasc"
    prompt = build_prompt(classes, prompt_tmpl)          # closed-ended prompt (with <image> cue)

    y_true, y_pred, raw_lines = [], [], []
    for rec in items:
        # 1) Load image in RGB (processor handles resize/normalize internally)
        img = open_rgb_image(rec["image"])

        # 2) Encode multimodal input
        inputs = processor(text=prompt, images=img, return_tensors="pt").to(model.device)

        # 3) Generate one short answer (closed-ended; no sampling by default)
        with np.errstate(all="ignore"):
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new,
                do_sample=temperature > 0.0,
                temperature=temperature if temperature > 0.0 else None,
            )

        text = processor.batch_decode(outputs, skip_special_tokens=True)[0]

        # 4) Parse to one of 7 labels (or None if hallucinated)
        pred = parse_to_label(text, classes) or "__other__"

        y_true.append(rec["answer"])
        y_pred.append(pred)
        raw_lines.append((rec["id"], rec["answer"], pred, text))

    # ------------------ Metrics -----------------------
    # Treat out-of-vocab as incorrect label "zzz" to keep scorers happy
    pred_for_metrics = [p if p in classes else "zzz" for p in y_pred]
    acc = accuracy_score(y_true, pred_for_metrics)
    f1 = f1_score(y_true, pred_for_metrics, labels=classes, average="macro")

    # Confusion matrix only for in-vocab predictions (to keep shape = 7x7)
    mask = [p in classes for p in y_pred]
    cm = confusion_matrix(np.array(y_true)[mask], np.array(y_pred)[mask], labels=classes)

    print(f"[VLM zero-shot] split={args.split}")
    print(f"Accuracy : {acc:.4f}")
    print(f"Macro-F1 : {f1:.4f}")
    print("Confusion (rows=true, cols=pred):")
    print(cm)



if __name__ == "__main__":
    main()
