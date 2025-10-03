import argparse
import json
from pathlib import Path
from typing import List
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import yaml
from src.visualize import  plot_training_curves, plot_confusion_matrix, plot_roc_curves, plot_per_class_report, plot_prob_histogram
from src.vqa_utils import parse_to_label

def eval_vlm(args):

    cfg = yaml.safe_load(open("configs.yaml"))
    classes: List[str] = cfg["data"]["classes"]                   

 
    questions = {}
    with open(args.true_answers_path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            questions[obj["question_id"]] = obj["answer"]  

    # Load model predictions
    preds = {}
    with open(args.pred_answers_path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            preds[obj["question_id"]] = obj["text"]

    # Build aligned lists
    y_true, y_pred = [], []
    for qid, ans in questions.items():
        if qid in preds:
            y_true.append(ans.strip())
            y_pred.append(preds[qid].strip())

    # ------------------ Evaluate ----------------------
    y_true, y_pred= [], []
    for qid, ans in questions.items():

        # Add true answer label
        y_true.append(ans.strip())

        # Add predicted text but 
        # parse it to one of 7 labels (or None if hallucinated)
        pred = parse_to_label(preds[qid], classes) or "__other__"

        y_pred.append(pred)
    
    print(y_true[:10])
    print(y_pred[:10])
    # ------------------ Metrics -----------------------
    # Treat out-of-vocab as incorrect label "zzz" to keep scorers ha
    pred_for_metrics = [p if p in classes else "zzz" for p in y_pred]
    acc = accuracy_score(y_true, pred_for_metrics)
    f1 = f1_score(y_true, pred_for_metrics, labels=classes, average="macro")

    # Confusion matrix only for in-vocab predictions (to keep shape = 7x7)
    mask = [p in classes for p in y_pred]
    cm = confusion_matrix(np.array(y_true)[mask], np.array(y_pred)[mask], labels=classes)

    # visualizations 
    plot_confusion_matrix(np.array(cm), classes,  "figures/baseline_confusion_eval_vlm.png")



    print(f"Accuracy : {acc:.4f}")
    print(f"Macro-F1 : {f1:.4f}")
    print("Confusion (rows=true, cols=pred):")
    print(cm)


def main():
    parser = argparse.ArgumentParser(description="Evaluate LLaVA model")
    parser.add_argument("--true_answers_path",  default="data/processed/vqa_prompt1/test.jsonl")
    parser.add_argument("--pred_answers_path",  default="results/finetune_checkpoint1000_results_prompt1.jsonl")
    
    args = parser.parse_args()
    eval_vlm(args)

if __name__ == "__main__":
    main()
