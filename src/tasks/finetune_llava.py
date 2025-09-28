import argparse
import subprocess, shlex
import os, yaml
from pathlib import Path

"""
LLaVA fine-tuning script
either: python tasks/finetune_llava.py --mode lora
        python tasks/finetune_llava.py --mode full
"""

def run_command(cmd):
    cmd_str = " ".join(shlex.quote(x) for x in cmd)
    try:
        result = subprocess.run(cmd_str, shell=True, check=True, capture_output=True, text=True)
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print("STDERR:\n", e.stderr)  
        print("STDOUT:\n", e.stdout)
        raise

    # if result.returncode != 0:
    #     print(f"Error: {result.stderr}")
    #     return False
    # return True

def finetune_llava(mode="lora"):
    """Fine-tune LLaVA model"""
    
    # Configuration
    cfg = yaml.safe_load(open("configs.yaml"))
    project_root = "/home/alaa.mohamed/VQA-VLM-Derma"
    llava_repo = "/home/alaa.mohamed/repos/LLaVA"  
    model_path = "microsoft/llava-med-v1.5-mistral-7b"
    data_path = f"{project_root}/data/processed/llava_format/train.json"
    image_folder = f"{project_root}/data/processed/images"
    output_dir = f"{project_root}/checkpoints/{mode}"
    
    # Set environment variables
    os.environ["TMPDIR"] = "/tmp"
    os.environ["HF_HOME"] = "/tmp/huggingface"
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    print(f"====================================")
    print(f"LLaVA Fine-tuning Mode: {mode}")
    print(f"====================================")
    # Common arguments
    base_cmd = [
            # "deepspeed", f"{llava_repo}/llava/train/train_mem.py",
            "deepspeed", "--num_gpus", "1",
            f"{llava_repo}/llava/train/train.py", 
            "--deepspeed", f"{llava_repo}/scripts/zero2.json",
            "--model_name_or_path", model_path,
            "--version", "v1",
            "--data_path", data_path,
            "--image_folder", image_folder,
            "--vision_tower", "openai/clip-vit-large-patch14-336",
            "--mm_projector_type", "mlp2x_gelu",
            "--mm_vision_select_layer", "-2",
            "--mm_use_im_start_end", 
            "--mm_use_im_patch_token", 
            "--image_aspect_ratio", "pad",
            "--group_by_modality_length",
            "--bf16",
            "--output_dir", output_dir,
            "--num_train_epochs", "3",
            "--evaluation_strategy", "no",
            "--save_strategy", "steps",
            "--save_steps", "300",
            "--save_total_limit", "2",
            "--weight_decay", "0.01",
            "--warmup_ratio", "0.03",
            "--lr_scheduler_type", "cosine",
            "--logging_steps", "10",
            "--model_max_length", "2048",
            "--gradient_checkpointing",
            "--dataloader_num_workers", "6",
            "--lazy_preprocess", 
            "--report_to", "none"
        ]
    
    
    if mode == "lora":
        print("LoRA Fine-tuning Selected")
        lora_args = [
            "--lora_enable", 
            "--lora_r", "16",
            "--lora_alpha", "32", 
            "--lora_dropout", "0.05",
            "--mm_projector_lr", "2e-5",
            "--per_device_train_batch_size", "1",
            "--gradient_accumulation_steps", "4",
            "--learning_rate", "2e-4"
        ]
        cmd_args = base_cmd + lora_args
        
    elif mode == "full":
        full_args = [
            "--per_device_train_batch_size", "6",
            "--gradient_accumulation_steps", "1", 
            "--learning_rate", "1e-5"
        ]
        cmd_args = base_cmd + full_args
    else:
        print(f"Invalid mode: {mode}")
        return False
    
    #  training
    success = run_command(cmd_args)
    
    if success:
        print(f"====================================")
        print(f"Training completed!")
        print(f"📁 Model saved to: {output_dir}")
        run_command("nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits")
        print(f"====================================")
    
    return success

def main():
    parser = argparse.ArgumentParser(description="Fine-tune LLaVA model")
    parser.add_argument("--mode", choices=["lora", "full"], default="lora", 
                       help="Fine-tuning mode (default: lora)")
    
    args = parser.parse_args()
    finetune_llava(args.mode)

if __name__ == "__main__":
    main()