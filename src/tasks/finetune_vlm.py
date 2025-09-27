
def collate_supervised_vqa(batch: list[dict], classes: list[str], prompt_template: str):
    """WHY: Supervised fine-tuning needs consistent inputs/targets.
       WHAT: For each record, build the same closed-ended prompt and set the target
             to the single canonical label token (teacher forcing).
       PITFALLS: Keep template IDENTICAL to zero-shot to isolate the effect of fine-tune."""

def train_lora(
    train_items: list[dict],
    val_items: list[dict],
    processor,
    model,
    classes: list[str],
    lora_cfg: dict,
    train_cfg: dict,
    prompt_template: str
) -> dict:
    """WHY: Encapsulate the fine-tuning loop.
       WHAT:
         - Inject LoRA with apply_lora_adapters().
         - Create dataloaders using collate_supervised_vqa().
         - Train for E epochs with small LR (LoRA), gradient accumulation if needed.
         - After each epoch, run vlm/eval.evaluate_split() on val_items.
         - Keep best adapters by val macro-F1; return {'best_state':..., 'history':...}.
       PITFALLS:
         - OOM? Use 4-bit loading + gradient checkpointing + small batch + accumulation.
         - Overfitting? Early-stop on val macro-F1; 1–3 epochs are typical for HAM10000."""

def load_adapters(model, state: dict):
    """WHY: Reuse best LoRA weights without touching the base.
       WHAT: Load adapter state dict into model; keep base frozen."""
