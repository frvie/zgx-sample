#!/usr/bin/env python3
"""
Download and prepare datasets for Llama fine-tuning.
This script downloads popular instruction-following datasets and formats them properly.
"""

import json
import os
from datasets import load_dataset
import argparse


def download_alpaca_dataset(output_path="./datasets/alpaca_dataset.json", max_samples=None):
    """
    Download and format the Stanford Alpaca dataset.
    This is one of the best datasets for instruction following.
    """
    print("Downloading Stanford Alpaca dataset...")
    
    try:
        # Load the dataset
        dataset = load_dataset("tatsu-lab/alpaca", split="train")
        print(f"Loaded {len(dataset)} samples from Alpaca dataset")
        
        # Limit samples if specified
        if max_samples and max_samples < len(dataset):
            dataset = dataset.select(range(max_samples))
            print(f"Limited to {max_samples} samples")
        
        # Convert to our format
        formatted_data = []
        for example in dataset:
            formatted_example = {
                "instruction": example["instruction"],
                "input": example.get("input", ""),
                "output": example["output"]
            }
            formatted_data.append(formatted_example)
        
        # Save to file
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(formatted_data, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Alpaca dataset saved to: {output_path}")
        return output_path
        
    except Exception as e:
        print(f"❌ Failed to download Alpaca dataset: {e}")
        return None


def download_dolly_dataset(output_path="./datasets/dolly_dataset.json", max_samples=None):
    """
    Download and format the Databricks Dolly dataset.
    High-quality instruction dataset created by Databricks.
    """
    print("Downloading Databricks Dolly dataset...")
    
    try:
        # Load the dataset
        dataset = load_dataset("databricks/databricks-dolly-15k", split="train")
        print(f"Loaded {len(dataset)} samples from Dolly dataset")
        
        # Limit samples if specified
        if max_samples and max_samples < len(dataset):
            dataset = dataset.select(range(max_samples))
            print(f"Limited to {max_samples} samples")
        
        # Convert to our format
        formatted_data = []
        for example in dataset:
            formatted_example = {
                "instruction": example["instruction"],
                "input": example.get("context", ""),  # Dolly uses "context" instead of "input"
                "output": example["response"]
            }
            formatted_data.append(formatted_example)
        
        # Save to file
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(formatted_data, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Dolly dataset saved to: {output_path}")
        return output_path
        
    except Exception as e:
        print(f"❌ Failed to download Dolly dataset: {e}")
        return None


def download_openassistant_dataset(output_path="./datasets/openassistant_dataset.json", max_samples=None):
    """
    Download and format the OpenAssistant dataset.
    Conversational dataset with high-quality human feedback.
    """
    print("Downloading OpenAssistant Guanaco dataset...")
    
    try:
        # Load the dataset
        dataset = load_dataset("timdettmers/openassistant-guanaco", split="train")
        print(f"Loaded {len(dataset)} samples from OpenAssistant dataset")
        
        # Limit samples if specified
        if max_samples and max_samples < len(dataset):
            dataset = dataset.select(range(max_samples))
            print(f"Limited to {max_samples} samples")
        
        # OpenAssistant format is already in text format, need to parse conversations
        formatted_data = []
        for example in dataset:
            text = example["text"]
            
            # Split into human and assistant parts
            if "### Human:" in text and "### Assistant:" in text:
                parts = text.split("### Assistant:")
                if len(parts) >= 2:
                    human_part = parts[0].replace("### Human:", "").strip()
                    assistant_part = parts[1].strip()
                    
                    formatted_example = {
                        "instruction": human_part,
                        "input": "",
                        "output": assistant_part
                    }
                    formatted_data.append(formatted_example)
        
        print(f"Processed {len(formatted_data)} valid conversations")
        
        # Save to file
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(formatted_data, f, indent=2, ensure_ascii=False)
        
        print(f"✅ OpenAssistant dataset saved to: {output_path}")
        return output_path
        
    except Exception as e:
        print(f"❌ Failed to download OpenAssistant dataset: {e}")
        return None


def create_mixed_dataset(datasets, output_path="./datasets/mixed_dataset.json", samples_per_dataset=1000):
    """
    Create a mixed dataset from multiple sources for more diverse training.
    """
    print("Creating mixed dataset...")
    
    all_data = []
    
    for dataset_path in datasets:
        if os.path.exists(dataset_path):
            with open(dataset_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            # Take a sample from each dataset
            sample_size = min(samples_per_dataset, len(data))
            sampled_data = data[:sample_size]
            all_data.extend(sampled_data)
            
            print(f"Added {len(sampled_data)} samples from {dataset_path}")
    
    # Shuffle the combined data
    import random
    random.seed(42)
    random.shuffle(all_data)
    
    # Save mixed dataset
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_data, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Mixed dataset with {len(all_data)} samples saved to: {output_path}")
    return output_path


def analyze_dataset(dataset_path):
    """Analyze dataset statistics."""
    print(f"\n📊 Analyzing dataset: {dataset_path}")
    
    with open(dataset_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"Total samples: {len(data)}")
    
    # Analyze instruction lengths
    instruction_lengths = [len(item["instruction"].split()) for item in data]
    output_lengths = [len(item["output"].split()) for item in data]
    
    print(f"Average instruction length: {sum(instruction_lengths) / len(instruction_lengths):.1f} words")
    print(f"Average output length: {sum(output_lengths) / len(output_lengths):.1f} words")
    print(f"Max instruction length: {max(instruction_lengths)} words")
    print(f"Max output length: {max(output_lengths)} words")
    
    # Show sample
    print(f"\n📝 Sample entry:")
    sample = data[0]
    print(f"Instruction: {sample['instruction'][:100]}...")
    if sample['input']:
        print(f"Input: {sample['input'][:100]}...")
    print(f"Output: {sample['output'][:100]}...")


def main():
    parser = argparse.ArgumentParser(description="Download and prepare datasets for fine-tuning")
    parser.add_argument("--dataset", type=str, choices=["alpaca", "dolly", "openassistant", "all"], 
                       default="alpaca", help="Dataset to download")
    parser.add_argument("--max_samples", type=int, default=None, 
                       help="Maximum number of samples to download")
    parser.add_argument("--output_dir", type=str, default="./datasets", 
                       help="Output directory for datasets")
    parser.add_argument("--create_mixed", action="store_true", 
                       help="Create a mixed dataset from all downloaded datasets")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Dataset Preparation for Llama Fine-tuning")
    print("=" * 60)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    downloaded_datasets = []
    
    if args.dataset == "alpaca" or args.dataset == "all":
        path = download_alpaca_dataset(
            os.path.join(args.output_dir, "alpaca_dataset.json"), 
            args.max_samples
        )
        if path:
            downloaded_datasets.append(path)
            analyze_dataset(path)
    
    if args.dataset == "dolly" or args.dataset == "all":
        path = download_dolly_dataset(
            os.path.join(args.output_dir, "dolly_dataset.json"), 
            args.max_samples
        )
        if path:
            downloaded_datasets.append(path)
            analyze_dataset(path)
    
    if args.dataset == "openassistant" or args.dataset == "all":
        path = download_openassistant_dataset(
            os.path.join(args.output_dir, "openassistant_dataset.json"), 
            args.max_samples
        )
        if path:
            downloaded_datasets.append(path)
            analyze_dataset(path)
    
    if args.create_mixed and len(downloaded_datasets) > 1:
        mixed_path = create_mixed_dataset(
            downloaded_datasets, 
            os.path.join(args.output_dir, "mixed_dataset.json"),
            samples_per_dataset=1000
        )
        if mixed_path:
            analyze_dataset(mixed_path)
    
    print(f"\n✅ Dataset preparation complete!")
    print(f"📁 Datasets saved in: {args.output_dir}")
    print(f"\n🚀 Ready for training! Use with:")
    for dataset_path in downloaded_datasets:
        print(f"   python finetune_tinyllama.py --dataset_path {dataset_path}")


if __name__ == "__main__":
    main()