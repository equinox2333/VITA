import torch
from datasets import Dataset
from transformers import Trainer, TrainingArguments, DataCollatorForSeq2Seq
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import os
import logging
from typing import Dict, List

from vita.rmm.models.memory_manager import ReasoningMemoryManager

class RMMTrainer:
    def __init__(
        self,
        model_name="Qwen/Qwen2.5-7B",
        output_dir="./rmm_output",
        training_args=None,
        lora_config=None
    ):
        # Initialize base model
        self.rmm = ReasoningMemoryManager(
            model_name=model_name,
            lora_r=lora_config.get("r", 16) if lora_config else 16,
            lora_alpha=lora_config.get("alpha", 32) if lora_config else 32,
            lora_dropout=lora_config.get("dropout", 0.05) if lora_config else 0.05
        )
        
        self.model = self.rmm.model
        self.tokenizer = self.rmm.tokenizer
        self.data_collator = DataCollatorForSeq2Seq(
            tokenizer=self.tokenizer, 
            padding=True,
            return_tensors="pt"
        )
        
        # Set up training arguments
        self.training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=3,
            per_device_train_batch_size=4,
            per_device_eval_batch_size=4,
            gradient_accumulation_steps=8,
            evaluation_strategy="steps",
            eval_steps=100,
            save_strategy="steps",
            save_steps=100,
            logging_dir=f"{output_dir}/logs",
            logging_steps=10,
            learning_rate=2e-4,
            weight_decay=0.01,
            warmup_steps=100,
            lr_scheduler_type="cosine",
            load_best_model_at_end=True,
            fp16=True,
            optim="paged_adamw_8bit",
            report_to="none",
            gradient_checkpointing=True
        )
        
        # Update with user provided arguments
        if training_args:
            for key, value in training_args.items():
                if hasattr(self.training_args, key):
                    setattr(self.training_args, key, value)
    
    def prepare_dataset(self, data: List[Dict]):
        """
        Prepare dataset for training
        
        Args:
            data: List of dictionaries containing 'reasoning', 'feedback', and 'memory_entry'
        """
        formatted_data = []
        
        for item in data:
            prompt = f"""Below is a reasoning process and feedback about errors in the reasoning. Create a structured memory entry that captures the key corrections and facts to remember.

### Reasoning:
{item['reasoning']}

### Feedback:
{item['feedback']}

### Memory Entry:"""
            
            inputs = self.tokenizer(prompt, truncation=True, padding=False)
            
            memory_entry = item['memory_entry']
            target = prompt + memory_entry
            labels = self.tokenizer(target, truncation=True, padding=False)
            
            # Create attention mask that masks the prompt part for loss calculation
            prompt_len = len(inputs["input_ids"])
            target_len = len(labels["input_ids"])
            
            # Set labels to -100 for prompt part (won't be included in loss calculation)
            labels_array = [-100] * prompt_len + labels["input_ids"][prompt_len:] 
            
            formatted_data.append({
                "input_ids": inputs["input_ids"],
                "attention_mask": inputs["attention_mask"],
                "labels": labels_array[:target_len]  # Truncate to match target length
            })
        
        return Dataset.from_list(formatted_data)
    
    def train(self, train_dataset, eval_dataset=None):
        """Train the memory manager"""
        trainer = Trainer(
            model=self.model,
            args=self.training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=self.data_collator
        )
        
        trainer.train()
        
        # Save trained model
        self.model.save_pretrained(self.training_args.output_dir)
        self.tokenizer.save_pretrained(self.training_args.output_dir)
        
        return trainer