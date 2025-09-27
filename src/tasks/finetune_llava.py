import argparse
import subprocess
import os, yaml
from pathlib import Path

"""
LLaVA fine-tuning script
either: python tasks/finetune_llava.py --mode lora
        python tasks/finetune_llava.py --mode full
"""

def run_command(cmd):
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error: {result.stderr}")
        return False
    return True

def finetune_llava(mode="lora"):
    """Fine-tune LLaVA model"""
    
    # Configuration
    cfg = yaml.safe_load(open("configs.yaml"))
    model_path = "microsoft/llava-med-v1.5-mistral-7b"
    data_path = "data/processed/llava_format/train.json"
    image_folder = "data/processed/images"
    output_dir = "checkpoints/{mode}"
    
    # Set environment variables
    os.environ["TMPDIR"] = "/tmp"
    os.environ["HF_HOME"] = "/tmp/huggingface"
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    print(f"====================================")
    print(f"LLaVA Fine-tuning Mode: {mode}")
    print(f"====================================")
    
    # Common arguments
    common_args = f"""
        --model_name_or_path {model_path}
        --version mistral_instruct
        --data_path {data_path}
        --image_folder {image_folder}
        --vision_tower openai/clip-vit-large-patch14-336
        --mm_projector_type mlp2x_gelu
        --mm_vision_select_layer -2
        --mm_use_im_start_end False
        --mm_use_im_patch_token False
        --image_aspect_ratio pad
        --group_by_modality_length True
        --bf16 True
        --output_dir {output_dir}
        --num_train_epochs 3
        --evaluation_strategy no
        --save_strategy steps
        --save_steps 300
        --save_total_limit 2
        --weight_decay 0.01
        --warmup_ratio 0.03
        --lr_scheduler_type cosine
        --logging_steps 10
        --tf32 True
        --model_max_length 2048
        --gradient_checkpointing True
        --dataloader_num_workers 6
        --lazy_preprocess True
        --report_to none
    """
    
    if mode == "lora":
        print("Starting LoRA training (~2-3 hours)..")
        cmd = f"""deepspeed {cfg["lava_repo_path"]}/llava/train/train_mem.py \\
            --lora_enable True \\
            --lora_r 256 \\
            --lora_alpha 512 \\
            --lora_dropout 0.05 \\
            --mm_projector_lr 2e-5 \\
            --deepspeed ./scripts/zero2.json \\
            --per_device_train_batch_size 4 \\
            --gradient_accumulation_steps 2 \\
            --learning_rate 2e-4 \\
            {common_args}"""
            
    elif mode == "full":
        print("Starting full fine-tuning (~6-8 hours)...")
        cmd = f"""deepspeed {cfg["lava_repo_path"]}/llava/train/train_mem.py \\
            --deepspeed ./scripts/zero2.json \\
            --per_device_train_batch_size 6 \\
            --gradient_accumulation_steps 1 \\
            --learning_rate 1e-5 \\
            {common_args}"""
    else:
        print(f"Invalid mode: {mode}")
        return False
    
    #  training
    success = run_command(cmd)
    
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