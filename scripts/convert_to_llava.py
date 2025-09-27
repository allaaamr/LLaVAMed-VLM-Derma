import json
import os
from pathlib import Path

"""
Convert VQA JSONL to LLaVA conversation format
"""

def convert_to_llava_format(input_file, output_file):
    
    llava_data = []
    
    with open(input_file, 'r') as f:
        for line in f:
            record = json.loads(line.strip())
            
            # Extract filename only
            image_name = Path(record["image"]).name
            
            # Create LLaVA conversation format
            llava_record = {
                "id": record["question_id"],
                "image": image_name,
                "conversations": [
                    {
                        "from": "human",
                        "value": f"<image>\n{record['text']}"
                    },
                    {
                        "from": "gpt",
                        "value": record["answer"]
                    }
                ]
            }
            
            llava_data.append(llava_record)
    
    # Save to JSON
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(llava_data, f, indent=2)
    
    print(f"✅ Converted {len(llava_data)} records: {input_file} → {output_file}")

def main():
    """Convert all VQA files to LLaVA format"""
    
    base_dir = "/home/alaa.mohamed/VQA-VLM-Derma"
    vqa_dir = f"{base_dir}/data/processed/vqa"
    llava_dir = f"{base_dir}/data/processed/llava_format"
    
    # Convert train, val, test
    for split in ["train", "val", "test"]:
        input_file = f"{vqa_dir}/{split}.jsonl"
        output_file = f"{llava_dir}/{split}.json"
        
        if os.path.exists(input_file):
            convert_to_llava_format(input_file, output_file)
        else:
            print(f"File not found: {input_file}")
    
    print(f"Data saved to: {llava_dir}/")

if __name__ == "__main__":
    main()