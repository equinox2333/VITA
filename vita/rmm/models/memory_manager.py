import os
import torch
from typing import List, Dict, Optional, Union, Any, Tuple
import logging

from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    BitsAndBytesConfig,
    GenerationConfig
)
from peft import (
    PeftModel, 
    PeftConfig, 
    LoraConfig, 
    get_peft_model, 
    prepare_model_for_kbit_training
)

from .memory_schema import MemoryEntry, MemoryBuffer


class ReasoningMemoryManager:
    """
    Reasoning Memory Manager (RMM) based on Qwen2.5 with LoRA.
    Manages memory entries based on feedback from reasoning processes.
    """
    
    def __init__(
        self,
        base_model_path: str,
        lora_adapter_path: Optional[str] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        load_in_8bit: bool = False,
        load_in_4bit: bool = True,
        use_flash_attention: bool = True,
        torch_dtype=torch.float16
    ):
        """
        Initialize the Reasoning Memory Manager.
        
        Args:
            base_model_path: Path to the Qwen2.5 base model
            lora_adapter_path: Path to the LoRA adapter weights
            device: Device to load the model on
            load_in_8bit: Whether to load the model in 8-bit precision
            load_in_4bit: Whether to load the model in 4-bit precision
            use_flash_attention: Whether to use flash attention
            torch_dtype: Torch data type for the model
        """
        self.logger = logging.getLogger(__name__)
        self.device = device
        self.base_model_path = base_model_path
        self.lora_adapter_path = lora_adapter_path
        
        # Configure quantization if needed
        quantization_config = None
        if load_in_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch_dtype,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
        elif load_in_8bit:
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True
            )
            
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            base_model_path,
            trust_remote_code=True,
            padding_side="right"
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load base model
        self.logger.info(f"Loading base model from {base_model_path}")
        self.model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            trust_remote_code=True,
            torch_dtype=torch_dtype,
            quantization_config=quantization_config,
            attn_implementation="flash_attention_2" if use_flash_attention else "eager",
            device_map="auto" if device == "cuda" else None
        )
        
        # Load LoRA adapter if provided
        if lora_adapter_path and os.path.exists(lora_adapter_path):
            self.logger.info(f"Loading LoRA adapter from {lora_adapter_path}")
            self.model = PeftModel.from_pretrained(
                self.model,
                lora_adapter_path,
                is_trainable=False  # Inference mode
            )
            
        # Set generation config
        self.generation_config = GenerationConfig(
            temperature=0.3,
            top_p=0.9,
            top_k=50,
            max_new_tokens=512,
            repetition_penalty=1.1
        )
        
        self.memory_buffer = MemoryBuffer()
    
    def format_prompt(
        self, 
        question: str,
        reasoning_steps: str,
        feedback: str,
        existing_memory: Optional[MemoryBuffer] = None
    ) -> str:
        """
        Format the input prompt for the memory manager.
        
        Args:
            question: Original question or problem
            reasoning_steps: Chain-of-thought reasoning steps
            feedback: Feedback on the reasoning
            existing_memory: Existing memory buffer, if any
            
        Returns:
            Formatted prompt string
        """
        memory_str = existing_memory.to_prompt_format() if existing_memory else ""
        
        prompt = f"""<|im_start|>system
You are a Reasoning Memory Manager tasked with maintaining and updating memory entries to improve reasoning.
Your job is to create concise, factual memory entries based on feedback about reasoning.
Format each memory entry as a correction, insight, or fact.
<|im_end|>

<|im_start|>user
# Question
{question}

# Reasoning Steps
{reasoning_steps}

# Feedback
{feedback}

{memory_str}

Based on the feedback, create new memory entries to guide future reasoning.
<|im_end|>

<|im_start|>assistant
"""
        return prompt
    
    def parse_memory_entries(self, response: str) -> List[MemoryEntry]:
        """
        Parse memory entries from the model's response.
        
        Args:
            response: Model's response text
            
        Returns:
            List of memory entries
        """
        entries = []
        
        # Clean up the response to extract just the memory entries
        if "<|im_end|>" in response:
            response = response.split("<|im_end|>")[0].strip()
        
        import re
        import uuid
        
        # Parse each entry based on common patterns
        entry_patterns = [
            r"\[CORRECTION\](.*?)(?=\[|\Z)",
            r"\[INSIGHT\](.*?)(?=\[|\Z)",
            r"\[FACT\](.*?)(?=\[|\Z)"
        ]
        
        entry_types = ["correction", "insight", "fact"]
        
        for pattern, entry_type in zip(entry_patterns, entry_types):
            matches = re.finditer(pattern, response, re.DOTALL)
            for match in matches:
                content = match.group(1).strip()
                if content:
                    entry = MemoryEntry(
                        id=str(uuid.uuid4()),
                        entry_type=entry_type,
                        content=content,
                        confidence=0.9,  # Default confidence
                        related_to=[],
                        metadata={}
                    )
                    entries.append(entry)
        
        # If no structured entries found but there's text, create a general insight
        if not entries and response.strip():
            entry = MemoryEntry(
                id=str(uuid.uuid4()),
                entry_type="insight",
                content=response.strip(),
                confidence=0.7,
                related_to=[],
                metadata={}
            )
            entries.append(entry)
            
        return entries
    
    def generate_memory_entries(
        self, 
        question: str, 
        reasoning_steps: str, 
        feedback: str,
        existing_memory: Optional[MemoryBuffer] = None
    ) -> List[MemoryEntry]:
        """
        Generate memory entries based on reasoning feedback.
        
        Args:
            question: Original question or problem
            reasoning_steps: Chain-of-thought reasoning steps
            feedback: Feedback on the reasoning
            existing_memory: Existing memory buffer, if any
            
        Returns:
            List of generated memory entries
        """
        prompt = self.format_prompt(question, reasoning_steps, feedback, existing_memory)
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        # Generate response
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                generation_config=self.generation_config,
                return_dict_in_generate=True,
                output_scores=False
            )
            
        response = self.tokenizer.decode(outputs.sequences[0], skip_special_tokens=False)
        
        # Extract assistant's response part
        if "<|im_start|>assistant" in response:
            response = response.split("<|im_start|>assistant")[1].strip()
        
        # Parse memory entries from the response
        memory_entries = self.parse_memory_entries(response)
        
        return memory_entries
    
    def update_memory(
        self, 
        question: str, 
        reasoning_steps: str, 
        feedback: str
    ) -> MemoryBuffer:
        """
        Update memory buffer with new entries based on feedback.
        
        Args:
            question: Original question or problem
            reasoning_steps: Chain-of-thought reasoning steps
            feedback: Feedback on the reasoning
            
        Returns:
            Updated memory buffer
        """
        new_entries = self.generate_memory_entries(
            question, 
            reasoning_steps, 
            feedback,
            self.memory_buffer
        )
        
        for entry in new_entries:
            self.memory_buffer.add_entry(entry)
            
        return self.memory_buffer
    
    def get_memory_buffer(self) -> MemoryBuffer:
        """Get the current memory buffer."""
        return self.memory_buffer
    
    def clear_memory(self) -> None:
        """Clear the memory buffer."""
        self.memory_buffer.clear()