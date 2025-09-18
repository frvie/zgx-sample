# Sample package for creating a UV virtual environment and Unsloth fine-tuning compatible scripts.

A streamlined toolkit for fine-tuning Llama-3.2-1B models using Unsloth for 2x faster training.

## 🛠️ Prerequisites## 📁 Project Structure

```
unsloth/
├── unsloth-finetune-llama3_2-1B.py    # Main training script
├── download_datasets.py               # Dataset preparation utility
├── gradio_chatbot.py                  # Gradio web interface for testing
├── streamlit_chatbot.py               # Streamlit web interface for testing
├── launch_chatbot.sh                  # Easy launcher for chatbot interfaces
├── datasets/                          # Training datasets
│   ├── alpaca_dataset.json
│   ├── dolly_dataset.json
│   └── mixed_dataset.json
├── pyproject.toml                      # Project dependencies and metadata
├── .venv/                             # UV virtual environment
└── README.md                          # This file
```

## ⚡ UV Package Manager Features

### Environment Management
```bash
# Create new environment
uv venv --python 3.12

# Activate environment
source .venv/bin/activate  # Linux/macOS
# or
.venv\Scripts\activate     # Windows

# Deactivate environment
deactivate
```

### Dependency Management
```bash
# Add new dependency
uv add package_name

# Add development dependency
uv add --dev pytest

# Remove dependency
uv remove package_name

# Update all dependencies
uv lock --upgrade

# Install from lock file
uv sync
```

### Project Commands
```bash
# Run script with UV
uv run python download_datasets.py

# Run with specific Python version
uv run --python 3.12 python unsloth-finetune-llama3_2-1B.py

# Show project info
uv info

# Export requirements
uv export --format requirements-txt > requirements.txt
```age Manager**: Fast Python package and project manager
- **CUDA 12.8+**: For GPU acceleration
- **RTX 5090** or similar high-end GPU
- **Python 3.12+**

## 🚀 Quick Start

### 1. Install UV Package Manager

If you don't have UV installed:

```bash
# Install UV (Linux/macOS)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Or with pip
pip install uv
```

### 2. Environment Setup with UV

#### Create New Virtual Environment from pyproject.toml

```bash
# Clone or navigate to the project directory
cd /path/to/unsloth

# Create virtual environment and install all dependencies
uv venv --python 3.12
source .venv/bin/activate

# Install project dependencies from pyproject.toml
uv pip install -e .

# Or install dependencies manually
uv pip install torch==2.8.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
uv pip install unsloth[cu128_ampere_flash_attn] transformers accelerate peft trl datasets
uv pip install scikit-learn matplotlib gradio streamlit
```

#### Alternative: Use UV Sync (Recommended)

```bash
# Create environment and sync with pyproject.toml in one command
uv sync

# Activate the environment
source .venv/bin/activate
```

#### Verify Installation

```bash
# Verify GPU and PyTorch setup
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"No GPU\"}')"

# Verify Unsloth installation
python -c "from unsloth import FastLanguageModel; print('✅ Unsloth installed successfully')"
```

### 3. Dataset Preparation
```bash
# Download and prepare datasets
python download_datasets.py

# Download all datasets and create mixed dataset
python download_datasets.py --dataset all --create_mixed
```

### 4. Training

#### Basic Training
```bash
python unsloth-finetune-llama3_2-1B.py \
  --dataset_path ./datasets/mixed_dataset.json \
  --max_steps 200 \
  --batch_size 2 \
  --learning_rate 2e-4 \
  --output_dir ./my_finetuned_model
```

#### Training with Validation
```bash
python unsloth-finetune-llama3_2-1B.py \
  --dataset_path ./datasets/mixed_dataset.json \
  --max_steps 200 \
  --batch_size 2 \
  --learning_rate 2e-4 \
  --enable_validation \
  --validation_ratio 0.1 \
  --eval_steps 25 \
  --save_plots \
  --output_dir ./my_finetuned_model
```

#### Resume Training
```bash
python unsloth-finetune-llama3_2-1B.py \
  --dataset_path ./datasets/mixed_dataset.json \
  --max_steps 200 \
  --resume_from_checkpoint ./my_finetuned_model/checkpoint-100 \
  --output_dir ./my_finetuned_model
```

### 5. Interactive Testing with Chatbot Interfaces

#### 🎨 Gradio Interface (Recommended)
```bash
# Launch Gradio chatbot with default model
./launch_chatbot.sh

# Launch with specific model and port
./launch_chatbot.sh -m ./llama32_mixed_finetuned -p 7860

# Create public sharing link
./launch_chatbot.sh -s
```

#### 📊 Streamlit Interface
```bash
# Launch Streamlit chatbot
./launch_chatbot.sh -i streamlit

# Custom port
./launch_chatbot.sh -i streamlit -p 8501
```

#### 🔧 Manual Testing
```python
from unsloth import FastLanguageModel
import torch

# Load your fine-tuned model
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="./llama32_mixed_finetuned",
    max_seq_length=2048,
    dtype=torch.bfloat16,
    load_in_4bit=True,
)

# Test with a prompt
FastLanguageModel.for_inference(model)
inputs = tokenizer(["### Instruction:\nWrite a story about AI\n\n### Response:\n"], return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=100)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

## � Project Structure

```
unsloth/
├── unsloth-finetune-llama3_2-1B.py    # Main training script
├── download_datasets.py               # Dataset preparation utility
├── datasets/                          # Training datasets
│   ├── alpaca_dataset.json
│   ├── dolly_dataset.json
│   └── mixed_dataset.json
├── pyproject.toml                      # Project dependencies
└── README.md                          # This file
```

## 🎛️ Training Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--model_name` | `unsloth/Llama-3.2-1B-Instruct` | Base model to fine-tune |
| `--dataset_path` | `None` | Path to training dataset JSON |
| `--output_dir` | `./llama32_finetuned` | Output directory for model |
| `--max_steps` | `100` | Maximum training steps |
| `--learning_rate` | `2e-4` | Learning rate |
| `--batch_size` | `2` | Per-device batch size |
| `--max_seq_length` | `2048` | Maximum sequence length |
| `--lora_rank` | `16` | LoRA adapter rank |

## 📊 Validation Features

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--enable_validation` | `False` | Enable validation during training |
| `--validation_ratio` | `0.1` | Fraction of data for validation |
| `--eval_steps` | `25` | Steps between evaluations |
| `--save_plots` | `False` | Generate training plots |

## 🎯 Datasets

Three pre-configured datasets are available:

- **Alpaca**: 52K instruction-following examples
- **Dolly**: 15K human-generated prompt/response pairs  
- **Mixed**: Combined Alpaca + Dolly (4K samples)

## � Performance

- **Training Speed**: 2x faster with Unsloth optimizations
- **Memory Usage**: ~74MB VRAM on RTX 5090
- **Model Size**: 1.24B parameters (0.9% trainable with LoRA)

## 🏆 Results

After 200 training steps:
- Training loss: 0.97 (60% reduction from start)
- Excellent instruction following
- Coherent creative and technical responses
- Professional communication capabilities

## 💡 Tips

1. **Start small**: Use `--max_steps 20` for quick testing
2. **Enable validation**: Use `--enable_validation --save_plots` for monitoring
3. **Resume training**: Save time with `--resume_from_checkpoint`
4. **GPU optimization**: Unsloth automatically optimizes for your hardware

## 🛠️ Dependencies

### Core Requirements (managed by UV)
- **PyTorch 2.8.0+** with CUDA 12.8 support
- **Unsloth** for fast fine-tuning
- **Transformers 4.55.0+** for model handling
- **Datasets, PEFT, TRL** for training pipeline
- **Validation**: scikit-learn, matplotlib
- **Interactive UI**: gradio, streamlit

### Installation Methods

#### Method 1: UV Sync (Recommended)
```bash
uv sync  # Installs everything from pyproject.toml
```

#### Method 2: Manual UV Installation
```bash
uv pip install torch==2.8.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
uv pip install unsloth[cu128_ampere_flash_attn]
uv pip install transformers accelerate peft trl datasets
uv pip install scikit-learn matplotlib gradio streamlit
```

#### Method 3: Traditional pip (fallback)
```bash
pip install -e .
```

## 📝 License

This project uses Unsloth's free license for educational and research purposes.
