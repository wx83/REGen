"""
Fine-tune Llama-3 model on instruction dataset using LoRA
"""

import argparse
import pandas as pd
import torch
from pathlib import Path
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import Dataset
from transformers import EarlyStoppingCallback


def main():
    parser = argparse.ArgumentParser(description="Fine-tune Llama-3 model on instruction dataset")
    parser.add_argument("csv_path", type=str, help="Path to the training CSV file")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for checkpoints")
    parser.add_argument("--train_split", type=float, default=0.9, help="Training data split (default: 0.9)")
    parser.add_argument("--model_name", type=str, default="mistralai/Mistral-7B-Instruct-v0.2",
                        help="Model name from HuggingFace Hub")
    parser.add_argument("--use_4bit", action="store_true", default=True,
                        help="Use 4-bit quantization for memory efficiency")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--max_length", type=int, default=512, help="Max sequence length")
    args = parser.parse_args()

    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Load dataset from CSV
    print(f"Loading dataset from {args.csv_path}...")
    df = pd.read_csv(args.csv_path)

    # Load model and tokenizer
    print(f"Loading {args.model_name}...")
    device_map = "auto"

    # 4-bit quantization configuration
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=args.use_4bit,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        quantization_config=bnb_config if args.use_4bit else None,
        device_map=device_map,
        trust_remote_code=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    # Prepare model for training with LoRA
    model = prepare_model_for_kbit_training(model)

    # LoRA configuration
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj", "k_proj"],  # Works for both Llama and Mistral
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Prepare dataset
    print(f"Preparing dataset with {len(df)} examples...")
    dataset = Dataset.from_pandas(df)

    def format_prompt(example):
        """Format instruction + input + output for causal language modeling"""
        instruction = example.get("Instruction", "")
        input_text = example.get("input", "")
        output_text = example.get("output", "")

        # Create prompt: instruction + input -> output
        prompt = f"{instruction} {input_text}\n\nOutput:\n{output_text}"
        return {"text": prompt}

    dataset = dataset.map(format_prompt, remove_columns=dataset.column_names)

    def tokenize_function(examples):
        """Tokenize the text"""
        tokenized = tokenizer(
            examples["text"],
            max_length=args.max_length,
            truncation=True,
            padding="max_length",
        )
        # Labels are the same as input_ids for causal language modeling
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized

    dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

    # Split into train and validation
    split_dataset = dataset.train_test_split(test_size=1 - args.train_split, seed=42)
    train_dataset = split_dataset["train"]
    eval_dataset = split_dataset["test"]

    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(eval_dataset)}")

    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        warmup_steps=100,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=50,  # Evaluate every 50 steps for better early stopping
        save_steps=50,  # Save checkpoint every 50 steps
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="none",
        optim="paged_adamw_32bit",
        bf16=torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8,
        max_grad_norm=0.3,
        weight_decay=0.001,
    )

    # Early stopping callback
    early_stop_callback = EarlyStoppingCallback(
        early_stopping_patience=3,  # Stop if no improvement for 3 evaluations
        early_stopping_threshold=0.0  # Stop when validation loss doesn't decrease
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        callbacks=[early_stop_callback],
    )

    print(f"Starting training... Checkpoints will be saved to {args.output_dir}")
    print("Using LoRA for efficient fine-tuning")
    print("Early stopping enabled: training will stop when validation loss stops improving")
    trainer.train()

    # Save final model
    print(f"\nSaving final model to {args.output_dir}/final_model")
    trainer.model.save_pretrained(f"{args.output_dir}/final_model")
    tokenizer.save_pretrained(f"{args.output_dir}/final_model")
    print("Training complete!")


if __name__ == "__main__":
    main()
