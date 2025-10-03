from __future__ import annotations
import argparse, json
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import pandas as pd
import yaml


"""
  Convert HAM10000 image+label splits into closed-ended VQA records that a VLM
  can train/evaluate on. This keeps the SAME splits as the
  baseline classifier for a fair comparison.

  - Reads CSVs from configs.yaml: paths.splits/{train,val,test}.csv
  - Uses canonical class order from configs.yaml: data.classes
  - Writes JSONL files: data/processed/vqa/{train,val,test}.jsonl
  - Each record has fields:
      {
        "id": <image_id>,
        "image": "data/processed/images/<file>.jpg",
        question": "You are a medical vision assistant. Answer with EXACTLY one label from: akiec, bcc, bkl, df, mel, nv, vasc. Do not output anything else. Just the single label. What is the lesion type?",
        "answer": "nv",
        "answer_idx": 5,
        "template_id": "lesion_7way_v1"
      }

OPTIONAL CONFIG:
  vqa:
    question_template: "You are a medical vision assistant. Answer with EXACTLY one label from: akiec, bcc, bkl, df, mel, nv, vasc. Do not output anything else. Just the single label. What is the lesion type?",
    out_dir: data/processed/vqa
"""

# ---------- Helpers ----------

def load_cfg(cfg_path: str) -> dict:
    """
    Reads YAML and normalizes optional keys used by this script.
    To Centralize config loading and provide safe defaults if optional fields are missing.
   
    """
    cfg = yaml.safe_load(open(cfg_path))
    # prefer vqa.out_dir if provided, else paths.proc/vqa, else data/processed/vqa
    out_dir = cfg["vqa"].get("out_dir")
    if out_dir is None:
        base = cfg.get("paths", {}).get("proc", "data/processed")
        out_dir = str(Path(base) / "vqa")
        cfg["vqa"]["out_dir"] = out_dir
    return cfg


def read_split_csv(csv_path: Path, classes: list[str], images_root: Path) -> pd.DataFrame:
    """
    expected input format
      CSV columns: id,label,lesion_id
        - id: e.g., "ISIC_0027419" 
        - label: one of classes (e.g., "bkl")
        - lesion_id: e.g., "HAM_0000118" (kept for analysis; not used for training)

    OUTPUT COLUMNS:
      id, image, label, label_idx, lesion_id
    """
    df = pd.read_csv(csv_path, dtype={"id": str, "label": str, "lesion_id": str})

    df["image"] = df["id"].astype(str).str.strip() + ".jpg"

    # 2) Map label -> label_idx using your canonical order
    class_to_idx = {c: i for i, c in enumerate(classes)}
    df["label_idx"] = df["label"].map(class_to_idx).astype("Int64")  # Int64 allows NA temporarily

    # 3) Drop rows with unknown label or missing file
    before = len(df)
    df = df[df["label_idx"].notna()].copy()
    df["label_idx"] = df["label_idx"].astype(int)
    # df = df[df["image"].map(lambda p: Path(p).exists())]
    dropped = before - len(df)
    if dropped > 0:
        print(f"[warn] Dropped {dropped} rows in '{csv_path.name}' (bad label or missing image).")

    # 4) Return needed columns 
    return df[["id", "image", "label", "label_idx", "lesion_id"]]

def build_record(row: pd.Series, classes: List[str], question: str) -> Dict:
    """
    Create a single closed-ended VQA item from one image row.
    Fills question text with option list and stores both string and index answers.
    """
    return {
        "question_id": row["id"],
        "image": row["image"],
        "text": question,
        "answer": row["label"],
        "answer_id": int(row["label_idx"]),
        "template_id": "lesion_7way_v1",
    }

def write_jsonl(items: List[Dict], out_path: Path) -> None:
    """
    Writes the list of dicts to disk.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")

def make_split_jsonl(split_name: str, split_csv_dir: Path, classes: List[str],
                     images_root: Path, out_dir: Path, q_template: str) -> int:
    """
    Do the conversion for one split (train/val/test).
    Read CSV → normalize → map to VQA records → write JSONL.
    RETURNS: number of items written (for logging).
    """
    csv_path = split_csv_dir / f"{split_name}.csv"
    df = read_split_csv(csv_path, classes, images_root)
    items = [build_record(r, classes, q_template) for _, r in df.iterrows()]
    out_path = out_dir / f"{split_name}.jsonl"
    write_jsonl(items, out_path)
    print(f"[ok] {split_name}: wrote {len(items)} items → {out_path}")
    return len(items)



def main():
    """
    Produces VQA JSONL files for train/val/test under vqa.out_dir.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs.yaml", help="Path to configs.yaml")
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    classes: List[str] = cfg["data"]["classes"]
    splits_dir = Path(cfg["paths"]["splits"])
    images_root = Path(cfg["paths"]["images"])
    out_dir = Path(cfg["vqa"]["out_dir"])
    question = cfg["vqa"]["question_template"]

    # Generate for all three splits
    total = 0
    for split in ["train", "val", "test"]:
        total += make_split_jsonl(split, splits_dir, classes, images_root, out_dir, question)

    # Final consistency hint
    if total == 0:
        print("[warn] No items written; check your split CSVs and image paths.")
    else:
        print(f"[done] Wrote {total} VQA items across splits to {out_dir.resolve()}")


if __name__ == "__main__":
    main()
