# VQA-VLM-Derma: Vision-Language Models for Dermatological Classification

## 📋 Overview 

This project investigates whether vision-language models can match or exceed task-specific CNN architectures for medical image classification. We evaluate:

- **CNN Baseline**: ResNet-18 with class-weighted cross-entropy
- **Zero-Shot VLM**: LLaVA-Med without fine-tuning
- **Fine-Tuned VLM**: LLaVA-Med adapted to dermatological domain using LoRA


## 🏗️ Project Structure

```
VQA-VLM-Derma/
├── data/
│   ├── raw/                    # HAM10000 raw dataset
│   ├── processed/
│   │   ├── images/                     # Preprocessed images (224×224)
│   │   ├── splits/                    # train/val/test CSVs
│   │   ├── llava_format/              # vqa_pairs using prompt 1 but formulated in converational format for training
│   │   ├── llava_format_prompt2/      # vqa_pairs using prompt 2 but formulated in converational format for training
│   │   └── vqa_prompt1/               # VQA-formatted JSONL files on prompt 1 (testing for prompt sensitivity)
│   │   └── vqa_prompt2/               # VQA-formatted JSONL files on prompt 2 (testing for prompt sensitivity)
│   │   └── vqa_prompt3/               # VQA-formatted JSONL files on prompt 3 (testing for prompt sensitivity)
├── figures/         #contains evaluation curves and figures
├── src/
│   ├── models/
│   │   ├── baseline.py        # baseline model
│   ├── tasks/
│   │   ├── finetune_llava.py  # script for finetuning llava med using lava training scripts
│   │   ├── train_baseline.py  # CNN training script
│   │   └── eval_vlm.py        # VLM evaluation script
│   |── data.py
│   |── visualize.py
│   |── vqa_utils.py
│   └── utils.py
├── scripts/
│   |── convert_to_llava.py      # script to convert VQA JSONL to LLaVA conversation format
│   |── vqa_pairs.py          # script to  convert HAM10000 image+label splits into closed-ended VQA records
│   |── stats.py                # script to explore dataset stats (e.g class imbalance & distribution
│   └── prep.py                # Data preprocessing pipeline to prepare HAM10000 dataset (Verifies metadata↔image mapping, Splits by lesion_id, image preprocessing)
├── results/
│   |── finetune_results/          # finetuned results
│   └── eval_results/          # zero shot results
├── configs.yaml               # Central configuration file
├── requirements.txt
└── README.md
```

## 🚀 Quick Start

### 1. Clone Repository and Dependencies

```bash
# Clone main repository
git clone https://github.com/allaaamr/VQA-VLM-Derma.git
cd VQA-VLM-Derma

# Clone LLaVA (required for base implementation)
git clone https://github.com/haotian-liu/LLaVA.git

# Clone LLaVA-Med (medical-specific VLM)
git clone https://github.com/microsoft/LLaVA-Med.git
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

**Note**: The `requirements.txt` includes installation instructions for both LLaVA and LLaVA-Med dependencies. but make sure to vuild the llava repos based on their github readme guidelines 

### 3. Configure Paths

Edit `configs.yaml` to set your data paths:

```yaml
paths:
  raw: data/raw                    # HAM10000 download location
  proc: data/processed             # Processed data output
  splits: data/processed/splits    # Train/val/test splits
  images: data/processed/images    # Resized images
  vqa: data/processed/vqa          # VQA-formatted files

data:
  classes: [akiec, bcc, bkl, df, mel, nv, vasc]  # 7 diagnostic categories
```

## 📊 Dataset Preparation

### Download HAM10000

Download the HAM10000 dataset from [Harvard Dataverse](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T) and place it in the `data/raw` directory as specified in `configs.yaml`.

The raw dataset should contain:
- `HAM10000_images_part_1/` and `HAM10000_images_part_2/` (images)
- `HAM10000_metadata.csv` (labels and lesion IDs)

### Preprocess Data

Run the preprocessing script to prepare the dataset for training:

```bash
python scripts/prep.py --raw data/raw --proc data/processed --size 224
```

**This script performs:**
- ✅ Metadata-to-image mapping verification
- ✅ Lesion-level stratified splitting (prevents data leakage)
- ✅ Image resizing to 224×224 for efficient training
- ✅ Generation of train/val/test CSVs with `(id, label, lesion_id)`

**Output structure:**
```
data/processed/
├── images/
│   └── ISIC_*.jpg              # All preprocessed images
└── splits/
    ├── train.csv               # 7,130 images (5,312 lesions)
    ├── val.csv                 # 890 images (664 lesions)
    └── test.csv                # 1,995 images (1,494 lesions)
```

## 🔬 Experiments

### Experiment 1: CNN Baseline

Train & Evaluate the ResNet-18 baseline with class-weighted cross-entropy:

```bash
python src/tasks/train_baseline.py
```

**Configuration options:**
- Dropout: 0.0 or 0.2
- Class weighting: None, capped (max 3.0), or uncapped
- Learning rate: 5e-4 (Adam optimizer)
- Batch size: 32
- Epochs: 20


### Experiment 2: Zero-Shot VLM Evaluation

#### Step 2a: Convert to VQA Format

Generate VQA-formatted JSONL files from the preprocessed splits:

```bash
python scripts/vqa_pairs.py
```

**Output:** `data/processed/vqa/{train,val,test}.jsonl`

Each record contains:
```json
{
  "id": "ISIC_0024306",
  "image": "data/processed/images/ISIC_0024306.jpg",
  "question": "You are a medical vision assistant. Answer with EXACTLY one label from: akiec, bcc, bkl, df, mel, nv, vasc. Do not output anything else. Just the single label. What is the lesion type?",
  "answer": "nv",
  "answer_idx": 5,
  "template_id": "lesion_7way_v1"
}
```

**Optional:** Customize the question template in `configs.yaml`:

```yaml
vqa:
  question_template: "Classify this skin lesion. Answer with only one label: akiec, bcc, bkl, df, mel, nv, or vasc."
  out_dir: data/processed/vqa
```

#### Step 2b: Generate VLM Predictions

Run zero-shot inference using LLaVA-Med:

```bash
python LLaVA-Med/llava/eval/model_vqa.py \
  --conv-mode mistral_instruct \
  --model-path microsoft/llava-med-v1.5-mistral-7b \
  --question-file data/processed/vqa/test.jsonl \
  --image-folder data/processed/images \
  --answers-file results/eval_results/llava-med_zeroshot_answers.jsonl \
  --temperature 0.0
```


#### Step 2c: Evaluate Predictions

Calculate accuracy and macro-F1 scores & Confusion matrix generation:

```bash
python -m src.tasks.eval_vlm \
  --true_answers_path data/processed/vqa/test.jsonl \
  --pred_answers_path results/eval_results/llava-med_zeroshot_answers.jsonl
```


### Experiment 3: Fine-Tuned VLM

#### Step 3a: Convert to VQA conversational format

```bash
python scripts/convert_to_llava.py
```

**Output:** `data/processed/llava_format/{train,val,test}.jsonl`

#### Step 3a: Fine-Tune LLaVA-Med

Fine-tune using LoRA on the HAM10000 training set:

```bash
bash src/tasks/finetune_llava.py
```

**Training configuration:**
- LoRA rank: 128, alpha: 256
- Learning rate: 2e-5
- Effective batch size: 16 (2 per device × 8 gradient accumulation)
- LoRA dropout: 0.05
- Training epochs: 3-5

**Note:** Fine-tuning requires ~40GB GPU memory. Adjust `per_device_train_batch_size` if needed.

#### Step 3b: Evaluate Fine-Tuned Model

Generate predictions using the fine-tuned checkpoint:

```bash
python LLaVA-Med/llava/eval/model_vqa.py \
  --conv-mode mistral_instruct \
  --model-path results/checkpoints/llava-med-finetuned/checkpoint-xxx \
  --question-file data/processed/vqa/test.jsonl \
  --image-folder data/processed/images \
  --answers-file results/eval_results/llava-med_finetuned_answers.jsonl \
  --temperature 0.0
```

Then evaluate:

```bash
python -m src.tasks.eval_vlm \
  --true_answers_path data/processed/vqa/test.jsonl \
  --pred_answers_path results/eval_results/llava-med_finetuned_answers.jsonl
```




**Hardware used:**
- GPU: NVIDIA A100 (40GB)
```
