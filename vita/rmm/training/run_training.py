import os
import argparse
import json
import torch
from datasets import Dataset
import logging

from vita.rmm.training.trainer import RMMTrainer
from vita.rmm.training.utils import set_seed, log_training_params, prepare_datasets

def parse_args():
    parser = argparse.ArgumentParser(description="Train Reasoning Memory Manager")
    
    # Model arguments
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-7B", 
                        help="Base model to use for RMM")
    parser.add_argument("--output_dir", type=str, default="./rmm_output",
                        help="Directory to save the model")
    
    # LoRA arguments
    parser.add_argument("--lora_r", type=int, default=16,
                        help="LoRA attention dimension")
    parser.add_argument("--lora_alpha", type=int, default=32,
                        help="LoRA alpha parameter")
    parser.add_argument("--lora_dropout", type=float, default=0.05,
                        help="LoRA dropout rate")
    
    # Training arguments
    parser.add_argument("--data_path", type=str, required=True,
                        help="Path to training data")
    parser.add_argument("--train_batch_size", type=int, default=4,
                        help="Training batch size per device")
    parser.add_argument("--eval_batch_size", type=int, default=4,
                        help="Evaluation batch size per device")
    parser.add_argument("--num_epochs", type=int, default=3,
                        help="Number of training epochs")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8,
                        help="Number of gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, default=2e-4,
                        help="Learning rate")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    
    return parser.parse_args()

def main():
    args = parse_args()
    set_seed(args.seed)
    
    # Set up logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger = logging.getLogger(__name__)
    
    # Initialize trainer
    trainer = RMMTrainer(
        model_name=args.model_name,
        output_dir=args.output_dir,
        training_args={
            "per_device_train_batch_size": args.train_batch_size,
            "per_device_eval_batch_size": args.eval_batch_size,
            "num_train_epochs": args.num_epochs,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "learning_rate": args.learning_rate
        },
        lora_config={
            "r": args.lora_r,
            "alpha": args.lora_alpha,
            "dropout": args.lora_dropout
        }
    )
    
    # Log training parameters
    log_training_params(vars(args), args.output_dir)
    
    # Load and prepare datasets
    logger.info(f"Loading data from {args.data_path}")
    train_data, eval_data = prepare_datasets(args.data_path, tokenizer=trainer.tokenizer)
    
    logger.info(f"Training data size: {len(train_data)}")
    logger.info(f"Evaluation data size: {len(eval_data)}")
    
    # Prepare datasets for training
    train_dataset = trainer.prepare_dataset(train_data)
    eval_dataset = trainer.prepare_dataset(eval_data) if eval_data else None
    
    # Train model
    logger.info("Starting training")
    trainer.train(train_dataset, eval_dataset)
    
    logger.info(f"Training complete. Model saved to {args.output_dir}")

if __name__ == "__main__":
    main()