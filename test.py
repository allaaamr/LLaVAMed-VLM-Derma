from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from llava.model import LlavaMistralForCausalLM

model_path = "/home/alaa.mohamed/VQA-VLM-Derma/checkpoints/llava-med_prompt2NL/checkpoint-1200"

try:
    # Try loading the model
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = LlavaMistralForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=False,use_flash_attention_2=False )
    print("Model loaded successfully")
except Exception as e:
    print(f"Error loading model: {e}")


