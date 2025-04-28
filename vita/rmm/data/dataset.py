import os
import json
import torch
import random
import argparse
from typing import Dict, List, Tuple, Optional, Union
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, PreTrainedTokenizer
from PIL import Image

class MemoryDataConfig:
   GSM8K_PATH = "data/gsm8k"
   MATHVISTA_PATH = "data/mathvista"
   MMLU_VISION_PATH = "data/mmlu_vision"
   
   DEFAULT_MAX_LENGTH = 2048
   DEFAULT_PAD_TOKEN = "<pad>"
   DEFAULT_BOS_TOKEN = "<s>"
   DEFAULT_EOS_TOKEN = "</s>"
   DEFAULT_UNK_TOKEN = "<unk>"
   
   MEMORY_ENTRY_TYPES = [
       "ARITHMETIC_ERROR",
       "LOGIC_ERROR",
       "VISUAL_MISINTERPRETATION",
       "FACTUAL_ERROR",
       "CONCEPT_ERROR"
   ]

class MemoryEntry:
   def __init__(
       self,
       error_type: str,
       correction: str,
       context: str,
       confidence: float = 1.0,
   ):
       self.error_type = error_type
       self.correction = correction
       self.context = context
       self.confidence = confidence
   
   def to_dict(self) -> Dict:
       return {
           "error_type": self.error_type,
           "correction": self.correction,
           "context": self.context,
           "confidence": self.confidence
       }
   
   @classmethod
   def from_dict(cls, data: Dict) -> 'MemoryEntry':
       return cls(
           error_type=data["error_type"],
           correction=data["correction"],
           context=data["context"],
           confidence=data.get("confidence", 1.0)
       )
   
   def format(self) -> str:
       return f"ERROR_TYPE: {self.error_type}\nCORRECTION: {self.correction}\nCONTEXT: {self.context}\nCONFIDENCE: {self.confidence}"

class RMMDataset(Dataset):
   def __init__(
       self,
       data_path: str,
       tokenizer: PreTrainedTokenizer,
       split: str = "train",
       max_length: int = MemoryDataConfig.DEFAULT_MAX_LENGTH,
       use_images: bool = True,
   ):
       self.data_path = data_path
       self.tokenizer = tokenizer
       self.split = split
       self.max_length = max_length
       self.use_images = use_images
       
       self.data = self._load_data()
       
   def _load_data(self) -> List[Dict]:
       filepath = os.path.join(self.data_path, f"{self.split}.json")
       if not os.path.exists(filepath):
           raise FileNotFoundError(f"Data file not found: {filepath}")
       
       with open(filepath, "r") as f:
           data = json.load(f)
           
       return data
   
   def __len__(self) -> int:
       return len(self.data)
   
   def __getitem__(self, idx: int) -> Dict:
       item = self.data[idx]
       
       problem = item["problem"]
       problem_text = problem["text"]
       
       image = None
       if "image_path" in problem and self.use_images:
           img_path = os.path.join(self.data_path, "images", problem["image_path"])
           if os.path.exists(img_path):
               image = Image.open(img_path).convert("RGB")
       
       initial_reasoning = item["initial_reasoning"]
       feedback = item["feedback"]
       memory_entries = [MemoryEntry.from_dict(m) for m in item["memory_entries"]]
       input_text = f"Problem: {problem_text}\n\nReasoning: {initial_reasoning}\n\nFeedback: {feedback}\n\nMemory Entry:"
       target_text = "\n".join([m.format() for m in memory_entries])
       
       tokenized_input = self.tokenizer(
           input_text,
           padding="max_length",
           truncation=True,
           max_length=self.max_length,
           return_tensors="pt"
       )
       
       tokenized_target = self.tokenizer(
           target_text,
           padding="max_length",
           truncation=True,
           max_length=self.max_length,
           return_tensors="pt"
       )
       
       labels = tokenized_target["input_ids"].clone()
       labels[labels == self.tokenizer.pad_token_id] = -100  # Ignore padding tokens
       
       return {
           "input_ids": tokenized_input["input_ids"].squeeze(),
           "attention_mask": tokenized_input["attention_mask"].squeeze(),
           "labels": labels.squeeze(),
           "problem_id": item["id"],
           "image": image,
           "memory_entries": [m.to_dict() for m in memory_entries]
       }

class GSM8KMemoryDataset(RMMDataset):
   def __init__(
       self,
       tokenizer: PreTrainedTokenizer,
       split: str = "train",
       max_length: int = MemoryDataConfig.DEFAULT_MAX_LENGTH,
   ):
       super().__init__(
           data_path=MemoryDataConfig.GSM8K_PATH,
           tokenizer=tokenizer,
           split=split,
           max_length=max_length,
           use_images=False,
       )

class MathVistaMemoryDataset(RMMDataset):
   def __init__(
       self,
       tokenizer: PreTrainedTokenizer,
       split: str = "train",
       max_length: int = MemoryDataConfig.DEFAULT_MAX_LENGTH,
   ):
       super().__init__(
           data_path=MemoryDataConfig.MATHVISTA_PATH,
           tokenizer=tokenizer,
           split=split,
           max_length=max_length,
           use_images=True,
       )

class MMLUVisionMemoryDataset(RMMDataset):
   def __init__(
       self,
       tokenizer: PreTrainedTokenizer,
       split: str = "train",
       max_length: int = MemoryDataConfig.DEFAULT_MAX_LENGTH,
   ):
       super().__init__(
           data_path=MemoryDataConfig.MMLU_VISION_PATH,
           tokenizer=tokenizer,
           split=split,
           max_length=max_length,
           use_images=True,
       )

class CombinedMemoryDataset(Dataset):
   def __init__(
       self,
       tokenizer: PreTrainedTokenizer,
       split: str = "train",
       max_length: int = MemoryDataConfig.DEFAULT_MAX_LENGTH,
       gsm8k_ratio: float = 0.4,
       mathvista_ratio: float = 0.3,
       mmlu_vision_ratio: float = 0.3,
   ):
       self.gsm8k_dataset = GSM8KMemoryDataset(tokenizer, split, max_length)
       self.mathvista_dataset = MathVistaMemoryDataset(tokenizer, split, max_length)
       self.mmlu_vision_dataset = MMLUVisionMemoryDataset(tokenizer, split, max_length)
       
       self.dataset_ratios = {
           "gsm8k": gsm8k_ratio,
           "mathvista": mathvista_ratio,
           "mmlu_vision": mmlu_vision_ratio
       }
       
       total_samples = min(
           len(self.gsm8k_dataset) / gsm8k_ratio,
           len(self.mathvista_dataset) / mathvista_ratio,
           len(self.mmlu_vision_dataset) / mmlu_vision_ratio
       )
       
       self.gsm8k_samples = int(total_samples * gsm8k_ratio)
       self.mathvista_samples = int(total_samples * mathvista_ratio)
       self.mmlu_vision_samples = int(total_samples * mmlu_vision_ratio)
       
       self.gsm8k_indices = list(range(len(self.gsm8k_dataset)))
       self.mathvista_indices = list(range(len(self.mathvista_dataset)))
       self.mmlu_vision_indices = list(range(len(self.mmlu_vision_dataset)))
       
       random.shuffle(self.gsm8k_indices)
       random.shuffle(self.mathvista_indices)
       random.shuffle(self.mmlu_vision_indices)
       
       self.gsm8k_indices = self.gsm8k_indices[:self.gsm8k_samples]
       self.mathvista_indices = self.mathvista_indices[:self.mathvista_samples]
       self.mmlu_vision_indices = self.mmlu_vision_indices[:self.mmlu_vision_samples]
       
       self.dataset_map = []
       for i in range(self.gsm8k_samples):
           self.dataset_map.append(("gsm8k", self.gsm8k_indices[i]))
       for i in range(self.mathvista_samples):
           self.dataset_map.append(("mathvista", self.mathvista_indices[i]))
       for i in range(self.mmlu_vision_samples):
           self.dataset_map.append(("mmlu_vision", self.mmlu_vision_indices[i]))
       
       random.shuffle(self.dataset_map)
   
   def __len__(self) -> int:
       return len(self.dataset_map)
   
   def __getitem__(self, idx: int) -> Dict:
       dataset_name, dataset_idx = self.dataset_map[idx]
       
       if dataset_name == "gsm8k":
           item = self.gsm8k_dataset[dataset_idx]
       elif dataset_name == "mathvista":
           item = self.mathvista_dataset[dataset_idx]
       else:  # mmlu_vision
           item = self.mmlu_vision_dataset[dataset_idx]
       
       item["dataset_source"] = dataset_name
       
       return item

def create_memory_data_collator(tokenizer: PreTrainedTokenizer):
   def collate_fn(batch):
       has_images = "image" in batch[0] and batch[0]["image"] is not None
       
       images = None
       if has_images:
           images = [item.pop("image") for item in batch]
           images = [img for img in images if img is not None]
           if not images:
               images = None
       
       input_ids = torch.stack([item["input_ids"] for item in batch])
       attention_mask = torch.stack([item["attention_mask"] for item in batch])
       labels = torch.stack([item["labels"] for item in batch])
       
       problem_ids = [item["problem_id"] for item in batch]
       memory_entries = [item["memory_entries"] for item in batch]
       dataset_sources = [item.get("dataset_source", "unknown") for item in batch]
       
       result = {
           "input_ids": input_ids,
           "attention_mask": attention_mask,
           "labels": labels,
           "problem_ids": problem_ids,
           "memory_entries": memory_entries,
           "dataset_sources": dataset_sources
       }
       
       if images:
           result["images"] = images
           
       return result
   
   return collate_fn

def get_memory_data_loader(
   dataset: Dataset,
   batch_size: int,
   tokenizer: PreTrainedTokenizer,
   shuffle: bool = True,
   num_workers: int = 4,
):
   return DataLoader(
       dataset,
       batch_size=batch_size,
       shuffle=shuffle,
       num_workers=num_workers,
       collate_fn=create_memory_data_collator(tokenizer)
   )