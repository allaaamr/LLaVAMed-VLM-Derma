"""
  Centralize image transforms so training uses mild augmentation while
  validation/test stay deterministic. Pulls options from configs.yaml.

  - build_transforms(cfg, split): returns a torchvision Compose for 'train' or eval
  - HamDataset: minimal Dataset that reads split CSVs and applies the transform
  - make_loaders(bs, cfg): builds train/val/test DataLoaders that share class order
"""

from pathlib import Path                   
from PIL import Image                       
import pandas as pd                       
from torch.utils.data import Dataset, DataLoader 
from torchvision import transforms         
import yaml                               

_cfg = yaml.safe_load(open("configs.yaml"))
CLASSES = _cfg["data"]["classes"]            # canonical class order
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}  # str label → int index


def build_transforms(cfg: dict, split: str) -> transforms.Compose:
    """
      Provide split-specific transforms. Apply augmentation ONLY to 'train'
      to reduce overfitting, while 'val'/'test' are resize+normalize only.

      Returns a torchvision Compose pipeline built from configs.yaml knobs.
    """
    # target side length in pixels, e.g., 224 → 224x224 square
    size = cfg["data"]["image_size"]
    # pull augmentation options; absent keys default to no-op
    aug = cfg.get("augment", {})

    # start with a deterministic resize so all images are same geometry
    t = [transforms.Resize((size, size))]

    if split == "train":
        # RandomRotation: small ±deg rotations preserve dermoscopic structure but add variety
        if aug.get("rotation_deg", 0) > 0:
            t.append(transforms.RandomRotation(degrees=aug["rotation_deg"]))

        # RandomAffine with translate only (degrees=0): gentle width/height shifts
        if aug.get("translate", 0) > 0:
            t.append(transforms.RandomAffine(degrees=0,
                                             translate=(aug["translate"], aug["translate"])))

        # ColorJitter (brightness only): ±5% brightness; avoids hue/sat shifts that can distort cues
        if aug.get("brightness", 0) > 0:
            t.append(transforms.ColorJitter(brightness=aug["brightness"]))

        # Horizontal flip: dermoscopy orientation is arbitrary; mild and safe
        if aug.get("hflip", False):
            t.append(transforms.RandomHorizontalFlip(p=0.5))

        # AVOID heavy aug like RandomErasing/Cutout or big color shifts
        # because they can obscure lesion structures or distort clinical color clues.

    # Convert to tensor (0–1) then normalize with ImageNet stats (sane defaults for ViTs/CNNs)
    t += [
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],   # mean
                             [0.229, 0.224, 0.225])   # std
    ]

    return transforms.Compose(t)

class HamDataset(Dataset):
    """
      Minimal dataset that reads a split CSV (id,label,lesion_id) and returns
      (image_tensor, class_index).
    """
    def __init__(self, split_csv: Path, images_dir: str , tfm: transforms.Compose):
        df = pd.read_csv(split_csv)
        # build absolute paths to resized images on disk
        self.paths = [str(Path(images_dir) / f"{iid}.jpg") for iid in df["id"]]
        # map string labels to integer indices based on CLASSES order
        self.labels = [CLASS_TO_IDX[l] for l in df["label"]]
        # store the transform pipeline to apply on each read
        self.tfm = tfm

    def __len__(self):
        # dataset size = number of image paths
        return len(self.paths)

    def __getitem__(self, i: int):
        # load image, force RGB (safety), apply transform, return with label index
        img = Image.open(self.paths[i]).convert("RGB")
        return self.tfm(img), self.labels[i]

def make_loaders(batch_size: int, cfg: dict):
    """
      Build train/val/test DataLoaders that share transforms and class order.
      This keeps training fair and evaluation deterministic.

      Returns (datasets_dict, dataloaders_dict, classes_list)
    """
    # resolve common paths from configs.yaml
    splits = Path(cfg["paths"]["splits"])
    images = cfg["paths"]["images"]

    # build train-time and eval-time transforms from config
    tfm_train = build_transforms(cfg, split="train")
    tfm_eval  = build_transforms(cfg, split="val")   # same for val/test (no aug)

    # construct datasets backed by the CSVs created in preprocessing
    ds = {
        "train": HamDataset(splits / "train.csv", images, tfm_train),
        "val":   HamDataset(splits / "val.csv",   images, tfm_eval),
        "test":  HamDataset(splits / "test.csv",  images, tfm_eval),
    }

    dl = {
        "train": DataLoader(ds["train"], batch_size=batch_size, shuffle=True,  num_workers=4, pin_memory=True),
        "val":   DataLoader(ds["val"],   batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True),
        "test":  DataLoader(ds["test"],  batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True),
    }
    return ds, dl, CLASSES
