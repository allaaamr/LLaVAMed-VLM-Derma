from typing import List, Dict, Optional
from pathlib import Path
import json
import re
import warnings
import numpy as np
from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration

"""
- build_prompt:       Fills the closed-ended prompt with the 7-class options.
- normalize_text:     Makes model outputs easy to parse (lowercase, strip punctuation).
- alias_map:          mapping from clinical phrases to the 7 labels.
- parse_to_label:     Parse the text to one of the 7 labels.
- target_from_record: Pull the gold label from a JSONL record.
- load_vlm_model:     Load LLaVA-Med (processor + model) with dtype/device options.
- apply_lora_adapters:Attach LoRA/QLoRA adapters, freezing base.
- load_split_items:   Read JSONL items.
- open_rgb_image:     Load images as RGB (what HF processors expect).
"""

#------- Prompt Building --------#



def normalize_text(s: str) -> str:
    """
    Robust to punctuation/casing/whitespace. 
    keep only info needed to decide the label.
    """
    s = s.lower().strip()
    s = re.sub(r"[^a-z0-9\s]", " ", s)   # remove punctuation/symbols
    s = re.sub(r"\s+", " ", s)           # collapse whitespace
    return s

def alias_map() -> Dict[str, str]:
    """
    VLMs may answer with full clinical names (“melanocytic nevus”).
    Therefore, Map common phrases to canonical tokens {'melanocytic nevus':'nv', ...}.
    """
    return {
        # akiec
        "actinic keratosis": "akiec",
        "actinic keratoses": "akiec",
        "intraepithelial carcinoma": "akiec",
        "bowen": "akiec",
        "bowen s": "akiec",

        # bcc
        "basal cell carcinoma": "bcc",

        # bkl
        "benign keratosis": "bkl",
        "seborrheic keratosis": "bkl",
        "solar lentigo": "bkl",

        # df
        "dermatofibroma": "df",

        # mel
        "melanoma": "mel",

        # nv
        "melanocytic nevus": "nv",
        "melanocytic nevi": "nv",
        "nevus": "nv",
        "nevi": "nv",

        # vasc
        "vascular lesion": "vasc",
        "hemangioma": "vasc",
    }

def parse_to_label(text: str, classes: List[str]) -> Optional[str]:
    """
    Centralize the coercion of free text into one of 7 labels.
    """
    print("before " , text)
    norm = normalize_text(text)
    tokens = norm.split()
    aliases = alias_map()

    # exact token present?
    for c in classes:
        if c in tokens:
            print(c)
            return c

    #  any alias phrase present?
    for phrase, canon in aliases.items():
        if phrase in norm:
            print(canon)
            return canon

    # substring fallback 
    for c in classes:
        if c in norm:
            print("fallback ", c)
            return c
    print("No match")
    # No match 
    return None

def target_from_record(rec: dict) -> str:
    """Fine-tuning needs a gold target string.
       thus return the canonical label token from the JSONL record (e.g., 'nv')."""
    
    if "answer" not in rec or not isinstance(rec["answer"], str):
        raise KeyError("Record missing a valid 'answer' string.")
    return rec["answer"]

#--------- Data VQA ----------#

def load_split_items(jsonl_path: "Path") -> List[Dict]:
    """ Standardize I/O. Read {id,image,question,options,answer,answer_idx}
       from JSONL and return a list of dicts."""
    items: List[Dict] = []
    jsonl_path = Path(jsonl_path)
    if not jsonl_path.exists():
        raise FileNotFoundError(f"JSONL file not found: {jsonl_path}")

    missing = 0
    with jsonl_path.open("r", encoding="utf-8") as f:
        for ln, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as e:
                warnings.warn(f"[{jsonl_path.name}:{ln}] JSON decode error: {e}")
                continue

            if "image" not in rec or "answer" not in rec:
                warnings.warn(f"[{jsonl_path.name}:{ln}] Missing 'image' or 'answer'; skipping.")
                continue

            # Drop if image file doesn't exist 
            if not Path(rec["image"]).exists():
                missing += 1
                continue

            items.append(rec)

    if missing:
        warnings.warn(f"[{jsonl_path.name}] Skipped {missing} items due to missing image files.")
    if not items:
        warnings.warn(f"[{jsonl_path.name}] No valid items found.")

    return items

def open_rgb_image(path: str):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Image not found: {p}")
    return Image.open(str(p)).convert("RGB")