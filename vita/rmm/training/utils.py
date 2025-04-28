import torch
import random
import numpy as np
import os
import json
from typing import List, Dict, Any
import logging
from datasets import Dataset

def set_seed(seed: int):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def log_training_params(args: Dict[str, Any], log_dir: str):
    """Log training parameters to file"""
    os.makedirs(log_dir, exist_ok=True)
    with open(os.path.join(log_dir, "training_params.json"), "w") as f:
        json.dump(args, f, indent=2)

def prepare_datasets(data_path: str, train_ratio=0.9, tokenizer=None):
    """
    Load and prepare datasets from file
    
    Args:
        data_path: Path to the data file (JSON format)
        train_ratio: Ratio of training data
        tokenizer: Tokenizer to use for tokenization
    
    Returns:
        Tuple of (train_dataset, eval_dataset)
    """
    with open(data_path, 'r') as f:
        data = json.load(f)
    
    # Shuffle data for random split
    random.shuffle(data)
    
    # Split data into train and eval
    split_idx = int(len(data) * train_ratio)
    train_data = data[:split_idx]
    eval_data = data[split_idx:]
    
    # Format datasets for model
    formatted_train = []
    formatted_eval = []
    
    for item in train_data:
        prompt = f"""Below is a reasoning process and feedback about errors in the reasoning. Create a structured memory entry that captures the key corrections and facts to remember.

### Reasoning:
{item['reasoning']}

### Feedback:
{item['feedback']}

### Memory Entry:"""
        
        target = prompt + item['memory_entry']
        
        if tokenizer:
            # Tokenize for length validation
            inputs = tokenizer(prompt, truncation=True, padding=False)
            labels = tokenizer(target, truncation=True, padding=False)
            
            # Skip if too long
            if len(labels["input_ids"]) > tokenizer.model_max_length:
                continue
        
        formatted_train.append({
            "reasoning": item["reasoning"],
            "feedback": item["feedback"],
            "memory_entry": item["memory_entry"]
        })
    
    for item in eval_data:
        if tokenizer:
            # Validate length
            prompt = f"""Below is a reasoning process and feedback about errors in the reasoning. Create a structured memory entry that captures the key corrections and facts to remember.

### Reasoning:
{item['reasoning']}

### Feedback:
{item['feedback']}

### Memory Entry:"""
            
            target = prompt + item['memory_entry']
            inputs = tokenizer(prompt, truncation=True, padding=False)
            labels = tokenizer(target, truncation=True, padding=False)
            
            # Skip if too long
            if len(labels["input_ids"]) > tokenizer.model_max_length:
                continue
        
        formatted_eval.append({
            "reasoning": item["reasoning"],
            "feedback": item["feedback"],
            "memory_entry": item["memory_entry"]
        })
    
    return formatted_train, formatted_eval

def save_predictions(predictions: List[str], references: List[str], output_file: str):
    """Save model predictions and references to file"""
    with open(output_file, 'w') as f:
        for pred, ref in zip(predictions, references):
            f.write(f"Prediction: {pred}\n\nReference: {ref}\n\n{'='*50}\n\n")