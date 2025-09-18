# Datasets Directory

This directory contains the training datasets for fine-tuning.

## Available Datasets

Run `python download_datasets.py` to download and prepare:

- **`alpaca_dataset.json`** - 52K instruction-following examples from Stanford Alpaca
- **`dolly_dataset.json`** - 15K human-generated prompt/response pairs from Databricks Dolly
- **`mixed_dataset.json`** - Combined dataset (4K samples) for quick training

## Dataset Format

All datasets use the Alpaca instruction format:

```json
[
  {
    "instruction": "Write a creative story about a robot learning to paint.",
    "input": "",
    "output": "A young robot named Zeta was eager to learn how to paint..."
  }
]
```

## Usage

```bash
# Download all datasets
python download_datasets.py

# Use in training
python unsloth-finetune-llama3_2-1B.py --dataset_path ./datasets/mixed_dataset.json
```