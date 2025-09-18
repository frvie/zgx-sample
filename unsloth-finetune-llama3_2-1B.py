#!/usr/bin/env python3
"""
Fine-tune Llama-3.2-1B model using Unsloth for efficient training.
This script demonstrates how to fine-tune Llama-3.2-1B on a custom dataset.
"""

import torch
from unsloth import FastLanguageModel
from unsloth import is_bfloat16_supported
from transformers import TrainingArguments, EvalPrediction
from trl import SFTTrainer
from datasets import Dataset
import json
import argparse
import os
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
import time


def create_validation_split(dataset: Dataset, validation_ratio: float = 0.1) -> Tuple[Dataset, Dataset]:
    """
    Split dataset into training and validation sets.
    
    Args:
        dataset: The full dataset
        validation_ratio: Fraction of data to use for validation
    
    Returns:
        train_dataset, val_dataset: Split datasets
    """
    dataset_size = len(dataset)
    val_size = int(dataset_size * validation_ratio)
    train_size = dataset_size - val_size
    
    # Shuffle and split
    shuffled_dataset = dataset.shuffle(seed=42)
    train_dataset = shuffled_dataset.select(range(train_size))
    val_dataset = shuffled_dataset.select(range(train_size, dataset_size))
    
    print(f"📊 Dataset split: {train_size} training, {val_size} validation samples")
    return train_dataset, val_dataset


def compute_validation_metrics(model, tokenizer, val_dataset: Dataset, max_samples: int = 20) -> Dict:
    """
    Compute validation metrics for the model.
    
    Args:
        model: The model to evaluate
        tokenizer: The tokenizer
        val_dataset: Validation dataset
        max_samples: Maximum number of samples to evaluate
    
    Returns:
        Dictionary containing validation metrics
    """
    print("🔍 Computing validation metrics...")
    
    # Enable inference mode
    FastLanguageModel.for_inference(model)
    
    # Select subset for evaluation
    eval_samples = min(max_samples, len(val_dataset))
    eval_dataset = val_dataset.select(range(eval_samples))
    
    response_lengths = []
    generation_times = []
    successful_generations = 0
    
    for i, example in enumerate(eval_dataset):
        if i % 5 == 0:
            print(f"  Evaluating sample {i+1}/{eval_samples}")
        
        # Extract instruction and expected response
        text = example["text"]
        if "### Response:" in text:
            prompt_part, expected_response = text.split("### Response:", 1)
            prompt = prompt_part + "### Response:"
            expected_response = expected_response.strip()
        else:
            continue
        
        # Tokenize input
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        # Generate response and measure time
        try:
            start_time = time.time()
            with torch.no_grad():
                generated = model.generate(
                    **inputs,
                    max_new_tokens=100,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id,
                )
            generation_time = time.time() - start_time
            generation_times.append(generation_time)
            
            # Decode and measure response length
            generated_text = tokenizer.decode(generated[0], skip_special_tokens=True)
            response = generated_text[len(prompt):].strip()
            response_lengths.append(len(response))
            successful_generations += 1
            
        except Exception as e:
            print(f"  ⚠️  Generation failed for sample {i+1}: {e}")
            continue
    
    # Compute metrics
    avg_response_length = np.mean(response_lengths) if response_lengths else 0
    avg_generation_time = np.mean(generation_times) if generation_times else 0
    
    metrics = {
        "eval_avg_response_length": avg_response_length,
        "eval_avg_generation_time": avg_generation_time,
        "eval_successful_generations": successful_generations,
        "eval_success_rate": successful_generations / eval_samples if eval_samples > 0 else 0,
        "eval_samples": eval_samples
    }
    
    print(f"✅ Validation metrics computed:")
    print(f"  � Avg Response Length: {avg_response_length:.1f} chars")
    print(f"  ⏱️  Avg Generation Time: {avg_generation_time:.2f}s")
    print(f"  ✅ Success Rate: {metrics['eval_success_rate']:.1%}")
    print(f"  📊 Successful Generations: {successful_generations}/{eval_samples}")
    
    return metrics


def save_training_plots(metrics_history: List[Dict], output_dir: str):
    """
    Save training and validation plots.
    """
    if not metrics_history:
        return
    
    # Extract metrics
    steps = [m.get("step", i) for i, m in enumerate(metrics_history)]
    train_losses = [m.get("train_loss") for m in metrics_history if m.get("train_loss")]
    eval_losses = [m.get("eval_loss") for m in metrics_history if m.get("eval_loss")]
    perplexities = [m.get("eval_perplexity") for m in metrics_history if m.get("eval_perplexity")]
    
    # Create plots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
    
    # Training loss
    if train_losses:
        ax1.plot(steps[:len(train_losses)], train_losses, 'b-', label='Training Loss')
        ax1.set_xlabel('Steps')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training Loss')
        ax1.legend()
    
    # Validation loss
    if eval_losses:
        eval_steps = [steps[i] for i, m in enumerate(metrics_history) if m.get("eval_loss")]
        ax2.plot(eval_steps, eval_losses, 'r-', label='Validation Loss')
        ax2.set_xlabel('Steps')
        ax2.set_ylabel('Loss')
        ax2.set_title('Validation Loss')
        ax2.legend()
    
    # Perplexity
    if perplexities:
        eval_steps = [steps[i] for i, m in enumerate(metrics_history) if m.get("eval_perplexity")]
        ax3.plot(eval_steps, perplexities, 'g-', label='Perplexity')
        ax3.set_xlabel('Steps')
        ax3.set_ylabel('Perplexity')
        ax3.set_title('Model Perplexity')
        ax3.legend()
    
    # Combined losses
    if train_losses and eval_losses:
        ax4.plot(steps[:len(train_losses)], train_losses, 'b-', label='Training Loss')
        eval_steps = [steps[i] for i, m in enumerate(metrics_history) if m.get("eval_loss")]
        ax4.plot(eval_steps, eval_losses, 'r-', label='Validation Loss')
        ax4.set_xlabel('Steps')
        ax4.set_ylabel('Loss')
        ax4.set_title('Training vs Validation Loss')
        ax4.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "training_metrics.png"), dpi=300)
    plt.close()
    
    print(f"📈 Training plots saved to {output_dir}/training_metrics.png")


def setup_training_arguments(output_dir="./results",
                           per_device_train_batch_size=2,
                           per_device_eval_batch_size=2,
                           gradient_accumulation_steps=4,
                           warmup_steps=10,
                           max_steps=100,
                           learning_rate=2e-4,
                           fp16=False,
                           bf16=True,
                           logging_steps=1,
                           eval_steps=25,
                           optim="adamw_8bit",
                           weight_decay=0.01,
                           lr_scheduler_type="linear",
                           seed=3407,
                           enable_evaluation=True):
    """
    Setup training arguments for the SFT trainer with validation support.
    """
    # Auto-detect bf16 support
    if not is_bfloat16_supported():
        bf16 = False
        fp16 = True
        print("bf16 not supported, using fp16")
    
    training_args = TrainingArguments(
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        warmup_steps=warmup_steps,
        max_steps=max_steps,
        learning_rate=learning_rate,
        fp16=fp16,
        bf16=bf16,
        logging_steps=logging_steps,
        eval_steps=eval_steps if enable_evaluation else None,
        optim=optim,
        weight_decay=weight_decay,
        lr_scheduler_type=lr_scheduler_type,
        seed=seed,
        output_dir=output_dir,
        report_to=None,  # Disable wandb/tensorboard
        save_strategy="steps",
        save_steps=50,
        eval_strategy="steps" if enable_evaluation else "no",
        load_best_model_at_end=True if enable_evaluation else False,
        metric_for_best_model="eval_loss" if enable_evaluation else None,
        greater_is_better=False if enable_evaluation else None,
    )
    
    return training_args


def load_model_and_tokenizer(model_name="unsloth/Llama-3.2-1B-Instruct", 
                            max_seq_length=2048, 
                            dtype=None, 
                            load_in_4bit=True):
    """
    Load and configure the Llama-3.2-1B model with Unsloth optimizations.
    
    Args:
        model_name: HuggingFace model name or path
        max_seq_length: Maximum sequence length for training
        dtype: Data type (None for auto-detection)
        load_in_4bit: Whether to load model in 4-bit quantization
    
    Returns:
        model, tokenizer: Loaded model and tokenizer
    """
    # Auto-detect optimal dtype
    if dtype is None:
        dtype = torch.bfloat16 if is_bfloat16_supported() else torch.float16
    
    print(f"Loading model: {model_name}")
    print(f"Using dtype: {dtype}")
    print(f"Max sequence length: {max_seq_length}")
    print(f"4-bit quantization: {load_in_4bit}")
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        dtype=dtype,
        load_in_4bit=load_in_4bit,
        # token="hf_...", # Use if you have access tokens
    )
    
    # Don't call for_inference here - it will be called later for generation
    
    return model, tokenizer


def setup_lora_adapters(model, 
                       r=16, 
                       target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                     "gate_proj", "up_proj", "down_proj"],
                       lora_alpha=16,
                       lora_dropout=0.0,
                       bias="none",
                       use_gradient_checkpointing="unsloth",
                       random_state=3407,
                       use_rslora=False,
                       loftq_config=None):
    """
    Configure LoRA (Low-Rank Adaptation) for efficient fine-tuning.
    
    Args:
        model: The loaded model
        r: LoRA rank
        target_modules: Modules to apply LoRA to
        lora_alpha: LoRA alpha parameter
        lora_dropout: LoRA dropout rate
        bias: Bias configuration
        use_gradient_checkpointing: Gradient checkpointing method
        random_state: Random seed
        use_rslora: Whether to use rsLoRA
        loftq_config: LoftQ configuration
    
    Returns:
        model: Model with LoRA adapters configured
    """
    print("Setting up LoRA adapters...")
    print(f"LoRA rank: {r}")
    print(f"Target modules: {target_modules}")
    print(f"LoRA alpha: {lora_alpha}")
    
    model = FastLanguageModel.get_peft_model(
        model,
        r=r,
        target_modules=target_modules,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias=bias,
        use_gradient_checkpointing=use_gradient_checkpointing,
        random_state=random_state,
        use_rslora=use_rslora,
        loftq_config=loftq_config,
    )
    
    return model


def format_instruction_dataset(example, tokenizer):
    """
    Format dataset examples for instruction-following fine-tuning.
    Expected format: {"instruction": "...", "input": "...", "output": "..."}
    """
    instruction = example.get("instruction", "")
    input_text = example.get("input", "")
    output = example.get("output", "")
    
    # Create instruction prompt
    if input_text:
        prompt = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n{output}"
    else:
        prompt = f"### Instruction:\n{instruction}\n\n### Response:\n{output}"
    
    return {"text": prompt}


def load_dataset_from_json(json_path, tokenizer, max_samples=None):
    """
    Load dataset from JSON file and format for training.
    
    Args:
        json_path: Path to JSON file
        tokenizer: Tokenizer for formatting
        max_samples: Maximum number of samples to use
    
    Returns:
        dataset: Formatted dataset
    """
    print(f"Loading dataset from: {json_path}")
    
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if max_samples:
        data = data[:max_samples]
        print(f"Using {len(data)} samples")
    
    # Format dataset
    formatted_data = []
    for example in data:
        formatted_example = format_instruction_dataset(example, tokenizer)
        formatted_data.append(formatted_example)
    
    dataset = Dataset.from_list(formatted_data)
    return dataset


def create_sample_dataset():
    """
    Create a sample dataset for demonstration purposes.
    """
    sample_data = [
        {
            "instruction": "Explain what machine learning is.",
            "input": "",
            "output": "Machine learning is a subset of artificial intelligence that enables computers to learn and make decisions from data without being explicitly programmed for every task."
        },
        {
            "instruction": "Translate the following English text to French.",
            "input": "Hello, how are you?",
            "output": "Bonjour, comment allez-vous?"
        },
        {
            "instruction": "Write a Python function to calculate factorial.",
            "input": "",
            "output": "def factorial(n):\n    if n == 0 or n == 1:\n        return 1\n    else:\n        return n * factorial(n - 1)"
        },
        {
            "instruction": "Summarize the benefits of exercise.",
            "input": "",
            "output": "Exercise provides numerous benefits including improved cardiovascular health, stronger muscles and bones, better mental health, weight management, and increased energy levels."
        }
    ]
    
    return sample_data


def setup_training_arguments(output_dir="./results",
                           per_device_train_batch_size=2,
                           per_device_eval_batch_size=2,
                           gradient_accumulation_steps=4,
                           warmup_steps=10,
                           max_steps=100,
                           learning_rate=2e-4,
                           fp16=False,
                           bf16=True,
                           logging_steps=1,
                           eval_steps=25,
                           optim="adamw_8bit",
                           weight_decay=0.01,
                           lr_scheduler_type="linear",
                           seed=3407,
                           enable_evaluation=True):
    """
    Setup training arguments for the SFT trainer.
    """
    # Auto-detect bf16 support
    if not is_bfloat16_supported():
        bf16 = False
        fp16 = True
        print("bf16 not supported, using fp16")
    
    training_args = TrainingArguments(
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        warmup_steps=warmup_steps,
        max_steps=max_steps,
        learning_rate=learning_rate,
        fp16=fp16,
        bf16=bf16,
        logging_steps=logging_steps,
        eval_steps=eval_steps if enable_evaluation else None,
        optim=optim,
        weight_decay=weight_decay,
        lr_scheduler_type=lr_scheduler_type,
        seed=seed,
        output_dir=output_dir,
        report_to=None,  # Disable wandb/tensorboard
        save_strategy="steps",
        save_steps=50,
        eval_strategy="steps" if enable_evaluation else "no",
        load_best_model_at_end=True if enable_evaluation else False,
        metric_for_best_model="eval_loss" if enable_evaluation else None,
        greater_is_better=False if enable_evaluation else None,
    )
    
    return training_args


def main():
    parser = argparse.ArgumentParser(description="Fine-tune Llama-3.2-1B with Unsloth")
    parser.add_argument("--model_name", type=str, 
                       default="unsloth/Llama-3.2-1B-Instruct",
                       help="Model name or path")
    parser.add_argument("--dataset_path", type=str, default=None,
                       help="Path to JSON dataset file")
    parser.add_argument("--output_dir", type=str, default="./llama32_finetuned",
                       help="Output directory for saved model")
    parser.add_argument("--max_steps", type=int, default=100,
                       help="Maximum training steps")
    parser.add_argument("--learning_rate", type=float, default=2e-4,
                       help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=2,
                       help="Per device training batch size")
    parser.add_argument("--max_seq_length", type=int, default=2048,
                       help="Maximum sequence length")
    parser.add_argument("--lora_rank", type=int, default=16,
                       help="LoRA rank")
    parser.add_argument("--use_sample_data", action="store_true",
                       help="Use sample data instead of loading from file")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None,
                       help="Path to checkpoint directory to resume training from")
    parser.add_argument("--enable_validation", action="store_true",
                       help="Enable validation during training")
    parser.add_argument("--validation_ratio", type=float, default=0.1,
                       help="Fraction of data to use for validation")
    parser.add_argument("--eval_steps", type=int, default=25,
                       help="Number of steps between evaluations")
    parser.add_argument("--save_plots", action="store_true",
                       help="Save training plots")
    
    args = parser.parse_args()
    
    print("=" * 50)
    print("Llama-3.2-1B Fine-tuning with Unsloth")
    print("=" * 50)
    
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(
        model_name=args.model_name,
        max_seq_length=args.max_seq_length
    )
    
    # Setup LoRA adapters
    model = setup_lora_adapters(model, r=args.lora_rank)
    
    # Load or create dataset
    if args.use_sample_data or args.dataset_path is None:
        print("Using sample dataset...")
        sample_data = create_sample_dataset()
        formatted_data = [format_instruction_dataset(ex, tokenizer) for ex in sample_data]
        dataset = Dataset.from_list(formatted_data)
    else:
        dataset = load_dataset_from_json(args.dataset_path, tokenizer)
    
    print(f"Dataset size: {len(dataset)}")
    print("Sample data:")
    print(dataset[0]["text"][:500] + "...")
    
    # Split dataset for validation if enabled
    train_dataset = dataset
    eval_dataset = None
    if args.enable_validation and len(dataset) > 10:
        train_dataset, eval_dataset = create_validation_split(dataset, args.validation_ratio)
    elif args.enable_validation:
        print("⚠️  Dataset too small for validation split, disabling validation")
        args.enable_validation = False
    
    # Setup training arguments
    training_args = setup_training_arguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        eval_steps=args.eval_steps,
        enable_evaluation=args.enable_validation
    )
    
    # Create trainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset if args.enable_validation else None,
        dataset_text_field="text",
        max_seq_length=args.max_seq_length,
        dataset_num_proc=2,
        packing=False,  # Can make training 5x faster for short sequences
        args=training_args,
    )
    
    print("\nStarting training...")
    print(f"Training for {args.max_steps} steps")
    
    # Start training (with optional resume)
    if args.resume_from_checkpoint:
        print(f"Resuming from checkpoint: {args.resume_from_checkpoint}")
        trainer_stats = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    else:
        trainer_stats = trainer.train()
    
    print("\nTraining completed!")
    print(f"Training loss: {trainer_stats.training_loss:.4f}")
    
    # Compute final validation metrics if enabled
    final_metrics = {}
    if args.enable_validation and eval_dataset:
        print("\n" + "="*50)
        print("FINAL VALIDATION METRICS")
        print("="*50)
        final_metrics = compute_validation_metrics(model, tokenizer, eval_dataset)
        
        # Save metrics to file
        metrics_file = os.path.join(args.output_dir, "validation_metrics.json")
        with open(metrics_file, 'w') as f:
            json.dump(final_metrics, f, indent=2)
        print(f"💾 Validation metrics saved to {metrics_file}")
    
    # Save training plots if enabled
    if args.save_plots:
        try:
            # Extract training history from trainer
            training_history = []
            if hasattr(trainer.state, 'log_history'):
                training_history = trainer.state.log_history
                training_history.append(final_metrics)
            
            save_training_plots(training_history, args.output_dir)
        except Exception as e:
            print(f"⚠️  Could not save plots: {e}")
    
    # Save model
    print(f"\nSaving model to {args.output_dir}")
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    
    print("Fine-tuning completed successfully!")
    
    # Test inference
    print("\n" + "=" * 50)
    print("Testing inference...")
    
    FastLanguageModel.for_inference(model)
    
    test_prompt = "### Instruction:\nExplain what artificial intelligence is.\n\n### Response:\n"
    inputs = tokenizer(test_prompt, return_tensors="pt").to("cuda")
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=128,
            use_cache=True,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Prompt: {test_prompt}")
    print(f"Response: {response[len(test_prompt):]}")


if __name__ == "__main__":
    main()