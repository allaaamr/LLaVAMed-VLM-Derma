from pathlib import Path
import warnings; warnings.filterwarnings("ignore")
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
import yaml


"""
The following is the preprocessing script to prepare HAM10000 dataset once. It:
     - Verifies metadata↔image mapping
     - Splits by lesion_id to prevent leakage
     - Resize images to a fixed size for faster training
     - Writes:
        data/processed/
        ├─ images/ISIC_*.jpg
        └─ splits/{train,val,test}.csv  (id,label,lesion_id)
To run the script on the raw data:
  python scripts/prep.py --raw data/raw --proc data/processed --size 224
"""

# --- load config and freeze classes from config (NO constants.py) ---
cfg = yaml.safe_load(open("configs.yaml"))
CLASSES = cfg["data"]["classes"]       # canonical label order from YAML

def verify_and_collect(raw_dir: Path) -> pd.DataFrame:
    """
    Verify that files/labels are consistent & produce a manifest we can reuse.
    Returns a DataFrame with the columns needed downstream: [id, label, lesion_id, path]
    """
    meta_csv = raw_dir / "HAM10000_metadata.csv"                  
    assert meta_csv.exists(), f"Missing metadata CSV at {meta_csv}"

    meta = pd.read_csv(meta_csv)                                   # load metadata
    meta = meta.rename(columns={"image_id": "id", "dx": "label"})  # normalize column names

    # ensure required columns are present
    for col in ["id", "label", "lesion_id"]:
        assert col in meta.columns, f"Metadata missing column: {col}"

    # image folders (dataset ships as two parts)
    part1 = raw_dir / "HAM10000_images_part_1"
    part2 = raw_dir / "HAM10000_images_part_2"
    assert part1.exists() or part2.exists(), "Image folders not found under data/raw"

    # resolve each image's absolute path (part_1 or fallback to part_2)
    def to_path(i):
        p1 = part1 / f"{i}.jpg"
        return str(p1) if p1.exists() else str((part2 / f"{i}.jpg"))

    meta["path"] = meta["id"].apply(to_path)                        # add a path column

    # verify all files exist on disk
    missing = [p for p in meta["path"] if not Path(p).exists()]
    assert len(missing) == 0, f"Missing {len(missing)} image files. Example: {missing[:3]}"

    # verify label set matches our canonical list
    unknown = set(meta["label"].unique()) - set(CLASSES)
    assert not unknown, f"Unknown labels present: {unknown}. Expected {CLASSES}"

    return meta[["id", "label", "lesion_id", "path"]].copy()       # compact manifest

def split_by_lesion(df: pd.DataFrame, seed: int = 42):
    """
    Multiple images can belong to the same lesion_id.
    Splitting by image would leak the same lesion across splits therefore optimistic results.

    Therefore this function:
      - Stratifies by label at the lesion level (80/10/10)
      - Maps lesion splits back to image rows
      - Returns dict of image-level DataFrames: {"train": df, "val": df, "test": df}
    """
    lesion_df = df[["lesion_id", "label"]].drop_duplicates()       # one row per lesion
    # sanity: each lesion should have a single label
    bad = lesion_df.groupby("lesion_id")["label"].nunique()
    assert bad[bad > 1].empty, "Found lesions with multiple labels (dataset inconsistency)."

    # split lesions: 80/20, then 10% of train → val
    lesion_train, lesion_test = train_test_split(
        lesion_df, test_size=0.20, stratify=lesion_df["label"], random_state=seed
    )
    lesion_train, lesion_val = train_test_split(
        lesion_train, test_size=0.1111, stratify=lesion_train["label"], random_state=seed
    )

    split_map = {
        "train": set(lesion_train["lesion_id"]),
        "val":   set(lesion_val["lesion_id"]),
        "test":  set(lesion_test["lesion_id"]),
    }

    splits = {}
    for name, lesions in split_map.items():
        part = df[df["lesion_id"].isin(lesions)].copy()            # select all images for those lesions
        splits[name] = part[["id", "label", "lesion_id", "path"]]  # keep required columns
    return splits

def resize_once(df_all: pd.DataFrame, proc_images_dir: Path, size: int = 224):
    """
    Writes resized RGB JPEG copies under data/processed/images/
    in order to speed up training and ensure all images are the same size on disk.
    """
    proc_images_dir.mkdir(parents=True, exist_ok=True)             # make output dir

    # iterate unique ids (avoid duplicate work if a lesion has multiple images)
    for _id, row in df_all.drop_duplicates("id").set_index("id").iterrows():
        src = Path(row["path"])                                    # original image path
        dst = proc_images_dir / f"{_id}.jpg"                       # destination resized path
        if dst.exists():                                           # skip if already done
            continue
        img = Image.open(src).convert("RGB").resize((size, size))  # open, force RGB, resize
        img.save(dst, quality=95)                                  # write JPEG (quality ~95)

def write_split_csvs(splits: dict, proc_splits_dir: Path):
    """
    It saves train/val/test CSVs with columns: id,label,lesion_id
    in order to persist the exact split membership so all models share the same data
    in order to ensure a fair comparison between models

    """
    proc_splits_dir.mkdir(parents=True, exist_ok=True)             # ensure folder exists
    for name, part in splits.items():
        part[["id", "label", "lesion_id"]].to_csv(                 # write minimal split info
            proc_splits_dir / f"{name}.csv", index=False
        )

def main(raw: str = None, proc: str = None, size: int = None):
    """
    
    Orchestrate the whole preprocessing step with paths from configs.yaml by default.
    
    Preprocessing steps by calling:
    verify_and_collect → split_by_lesion → resize → write CSV splits; prints counts.
    """
    # resolve paths from CLI or configs.yaml
    raw_dir = Path(raw or cfg["paths"]["raw"])
    proc_dir = Path(proc or cfg["paths"]["proc"])
    images_out = proc_dir / "images"
    splits_out = proc_dir / "splits"
    target_size = size or cfg["data"]["image_size"]

    df = verify_and_collect(raw_dir)                               # integrity + manifest
    splits = split_by_lesion(df, seed=cfg["seed"])                 # lesion-level fair splits
    resize_once(df, images_out, size=target_size)                  # resize once for all ids
    write_split_csvs(splits, splits_out)                           # persist split CSVs

    #  readable summary
    for name, part in splits.items():
        print(f"{name:5s}: {len(part):5d} images | lesions={part['lesion_id'].nunique():4d}")
    print(f"resized images → {images_out}")
    print(f"splits         → {splits_out}")

if __name__ == "__main__":
    # to override paths/size in yaml file if needed
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw",  default=None)
    ap.add_argument("--proc", default=None)
    ap.add_argument("--size", type=int, default=None)
    args = ap.parse_args()
    main(args.raw, args.proc, args.size)
