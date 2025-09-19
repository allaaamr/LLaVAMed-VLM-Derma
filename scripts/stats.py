
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import yaml

"""
  This class is to inspect class balance. This function reads the processed splits (train/val/test),
  tallies how many images belong to each diagnostic class, prints the counts, and visualizes
  them as a bar chart (optionally grouped by split).

  - show_class_distribution(proc="data/processed", per_split=False, save_to=None, show=True)
    • If per_split=False: one bar per class (sum of train+val+test)
    • If per_split=True: grouped bars per class (train vs val vs test)
    • Prints a neat text table and returns a pandas DataFrame of counts

 To run: 
  python scripts/stats.py
"""


def show_class_distribution(proc: str = "data/processed",
                            per_split: bool = False,
                            save_to: str = "figures/unnamed.png",
                            show: bool = True) -> pd.DataFrame:
    """
      - Reads train/val/test CSVs from proc/splits/
      - Uses class order from configs.yaml
      - Prints a table of counts
      - Plots a bar chart (overall) or grouped chart (by split)
      - Optionally saves the figure to `save_to`
      - Returns a DataFrame of counts (index=class, columns=['count'] or ['train','val','test'])

    Parameters:
      proc (str): Folder where processed data lives (expects {proc}/splits/*.csv, {proc}/images/)
      per_split (bool): If True, show grouped bars by split; else show overall totals
      save_to (str|None): If set, path to write the figure (e.g., 'figures/class_dist.png')
      show (bool): If True, display the figure window (use False on headless servers)

    Returns:
      pandas.DataFrame: counts per class (and per split if requested)
    """

    # --- load config to get canonical class order  ---
    cfg_path = Path("configs.yaml") 
    cfg = yaml.safe_load(cfg_path.read_text())
    classes = cfg["data"]["classes"]

    # --- locate split CSVs ---
    splits_dir = Path(proc) / "splits"  # where train/val/test CSVs were written
    csv_paths = {
        "train": splits_dir / "train.csv",
        "val":   splits_dir / "val.csv",
        "test":  splits_dir / "test.csv",
    }
    # assert all three CSV files exist to avoid silent failures
    for name, p in csv_paths.items():
        if not p.exists():
            raise FileNotFoundError(f"Missing split file: {p}. Did you run scripts/prep.py?")

    # --- read split CSVs into DataFrames ---
    df_train = pd.read_csv(csv_paths["train"])
    df_val   = pd.read_csv(csv_paths["val"])
    df_test  = pd.read_csv(csv_paths["test"])

    # --- compute counts ---
    if per_split:
        # compute value counts for each split separately (reindex to include zero-count classes)
        c_train = df_train["label"].value_counts().reindex(classes, fill_value=0)
        c_val   = df_val["label"].value_counts().reindex(classes, fill_value=0)
        c_test  = df_test["label"].value_counts().reindex(classes, fill_value=0)
        # assemble into a single DataFrame with columns per split
        counts_df = pd.DataFrame({"train": c_train, "val": c_val, "test": c_test})
    else:
        # concatenate all splits if we want overall totals
        df_all = pd.concat([df_train, df_val, df_test], ignore_index=True)
        # count labels across the whole dataset and align order to `classes`
        counts = df_all["label"].value_counts().reindex(classes, fill_value=0)
        counts_df = counts.to_frame(name="count")

    print("\n# Images per class")
    print(counts_df)

    # --- plot ---
    plt.figure(figsize=(10, 4))
    if per_split:
        n = len(classes)
        x = range(n)
        width = 0.25
        plt.bar([i - width for i in x], counts_df["train"].values, width, label="train")
        plt.bar(x, counts_df["val"].values, width, label="val")
        plt.bar([i + width for i in x], counts_df["test"].values, width, label="test")
        plt.xticks(list(x), classes, rotation=0)
        plt.legend()
        plt.title("Images per class (by split)")
    else:
        # draw one bar per class using overall counts
        plt.bar(classes, counts_df["count"].values)
        # add the exact count above each bar for readability
        for i, v in enumerate(counts_df["count"].values):
            plt.text(i, v + max(counts_df["count"].values) * 0.01, str(int(v)), ha="center", va="bottom")
        plt.title("Images per class (overall)")

    plt.ylabel("Image count")
    plt.tight_layout()

    if save_to is not None:
        Path(save_to).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_to, dpi=200)
        print(f"\nFigure saved to: {save_to}")

    if show:
        plt.show()
    else:
        plt.close()

    return counts_df

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Show and plot image counts per class.")
    ap.add_argument("--proc", default="data/processed", help="Processed data folder (expects splits/)")
    ap.add_argument("--per-split", action="store_true", help="Group bars by split (train/val/test)")
    ap.add_argument("--save-to", default=None, help="Optional path to save the plot image")
    ap.add_argument("--no-show", action="store_true", help="Do not display the figure window")
    args = ap.parse_args()
    show_class_distribution(proc=args.proc, per_split=args.per_split, save_to=args.save_to, show=not args.no_show)
