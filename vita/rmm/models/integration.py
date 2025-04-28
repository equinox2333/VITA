import os
import torch
import copy
from typing import List, Dict, Optional, Union, Any, Tuple
import logging

from vita.model import VITAMixtralForCausalLM, VITAMistralForCausalLM, VITAQwen2ForCausalLM
from vita.conversation import conv_templates
from vita.constants import DEFAULT_IMAGE_TOKEN

from transformers import AutoTokenizer, AutoProcessor, AutoModelForCausalLM, BitsAndBytesConfig
from PIL import Image
import numpy as np

from .memory_manager import ReasoningMemoryManager
from .memory_schema import MemoryBuffer, MemoryEntry


class VITAWithMemory:
    """
    Integration of VITA model with Reasoning Memory Manager (RMM)
    for iterative, feedback-aware reasoning.
    """
    
    def __init__(
        self,
        vita_model_path: str,
        vita_model_type: str,
        rmm_base_model_path: str,
        rmm_lora_adapter_path: str,
        vision_tower_path: Optional[str] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        torch_dtype = torch.float16,
        load_in_8bit: bool = False,
        load_in_4bit: bool = True,
        use_flash_attention: bool = True,
        max_iterations: int = 3
    ):
        """
        Initialize VITA with Memory.
        
        Args:
            vita_model_path: Path to the VITA model
            vita_model_type: Type of VITA model (mixtral-8x7b, nemo, qwen2p5_instruct)
            rmm_base_model_path: Path to the RMM base model (Qwen2.5)
            rmm_lora_adapter_path: Path to the RMM LoRA adapter
            vision_tower_path: Path to the vision tower (if different from VITA's default)
            device: Device to load the models on
            torch_dtype: Torch data type for the models
            load_in_8bit: Whether to load models in 8-bit precision
            load_in_4bit: Whether to load models in 4-bit precision
            use_flash_attention: Whether to use flash attention
            max_iterations: Maximum number of reasoning iterations
        """
        self.logger = logging.getLogger(__name__)
        self.device = device
        self.torch_dtype = torch_dtype
        self.max_iterations = max_iterations
        
        # Quantization config
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
            
        # Initialize VITA model
        self.logger.info(f"Loading VITA model from {vita_model_path}")
        self.vita_model_type = vita_model_type
        
        # Load VITA tokenizer
        self.vita_tokenizer = AutoTokenizer.from_pretrained(
            vita_model_path,
            trust_remote_code=True,
            padding_side="right",
            use_fast=True
        )
        
        if self.vita_tokenizer.pad_token is None:
            self.vita_tokenizer.pad_token = self.vita_tokenizer.eos_token
        
        # Load VITA model based on model type
        if vita_model_type == "mixtral-8x7b":
            self.vita_model = VITAMixtralForCausalLM.from_pretrained(
                vita_model_path,
                trust_remote_code=True,
                torch_dtype=torch_dtype,
                quantization_config=quantization_config,
                attn_implementation="flash_attention_2" if use_flash_attention else "eager",
                device_map="auto" if device == "cuda" else None
            )
        elif vita_model_type == "nemo":
            self.vita_model = VITAMistralForCausalLM.from_pretrained(
                vita_model_path,
                trust_remote_code=True,
                torch_dtype=torch_dtype,
                quantization_config=quantization_config,
                attn_implementation="flash_attention_2" if use_flash_attention else "eager",
                device_map="auto" if device == "cuda" else None
            )
        elif vita_model_type == "qwen2p5_instruct":
            self.vita_model = VITAQwen2ForCausalLM.from_pretrained(
                vita_model_path,
                trust_remote_code=True,
                torch_dtype=torch_dtype,
                quantization_config=quantization_config,
                attn_implementation="flash_attention_2" if use_flash_attention else "eager",
                device_map="auto" if device == "cuda" else None
            )
        else:
            raise ValueError(f"Unsupported VITA model type: {vita_model_type}")
        
        # Initialize vision components
        self.logger.info("Initializing vision components")
        
        # Prepare vision encoder
        if vision_tower_path:
            from transformers import PretrainedConfig
            
            # Create a custom config with the vision tower path
            config = PretrainedConfig()
            config.vision_tower = vision_tower_path
            config.mm_projector_type = "mlp2x_gelu"
            
            # Initialize the vision encoder with the custom config
            self.vita_model.get_model().initialize_vision_modules(model_args=config)
        else:
            # Use default vision encoder
            from transformers import PretrainedConfig
            config = PretrainedConfig()
            config.vision_tower = self.vita_model.config.vision_tower
            config.mm_projector_type = "mlp2x_gelu"
            self.vita_model.get_model().initialize_vision_modules(model_args=config)
        
        # Get vision processor
        self.vision_tower = self.vita_model.get_vision_tower()
        self.image_processor = self.vision_tower.image_processor
        
        # Set the model to eval mode
        self.vita_model.eval()
        
        # Initialize RMM
        self.logger.info(f"Initializing Reasoning Memory Manager")
        self.rmm = ReasoningMemoryManager(
            base_model_path=rmm_base_model_path,
            lora_adapter_path=rmm_lora_adapter_path,
            device=device,
            load_in_8bit=load_in_8bit,
            load_in_4bit=load_in_4bit,
            use_flash_attention=use_flash_attention,
            torch_dtype=torch_dtype
        )
        
        # Set up conversation template
        self.conv_template = conv_templates["mixtral_two"]  # Default template
        if vita_model_type == "nemo":
            self.conv_template = conv_templates["nemo"]
        elif vita_model_type == "qwen2p5_instruct":
            self.conv_template = conv_templates["qwen2p5_instruct"]
    
    def process_image(self, image):
        """
        Process an image for input to the VITA model.
        
        Args:
            image: PIL image or path to image
            
        Returns:
            Processed image tensor
        """
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
            
        # Apply padding if needed
        if self.vita_model.config.image_aspect_ratio == "pad":
            def expand2square(pil_img, background_color):
                width, height = pil_img.size
                if width == height:
                    return pil_img
                elif width > height:
                    result = Image.new(pil_img.mode, (width, width), background_color)
                    result.paste(pil_img, (0, (width - height) // 2))
                    return result
                else:
                    result = Image.new(pil_img.mode, (height, height), background_color)
                    result.paste(pil_img, ((height - width) // 2, 0))
                    return result
                
            image = expand2square(image, tuple(int(x * 255) for x in self.image_processor.image_mean))
        
        # Process the image
        processed_image = self.image_processor(image, return_tensors="pt")["pixel_values"][0]
        
        return processed_image
    
    def generate_reasoning(
        self, 
        question: str, 
        image: Optional[Union[str, Image.Image]] = None,
        memory_buffer: Optional[MemoryBuffer] = None,
        use_cot: bool = True,
        max_new_tokens: int = 1024,
        temperature: float = 0.3,
        top_p: float = 0.9,
        do_sample: bool = True
    ) -> str:
        """
        Generate reasoning output from VITA model with memory.
        
        Args:
            question: Question or problem to solve
            image: Optional image (path or PIL image)
            memory_buffer: Optional memory buffer
            use_cot: Whether to use chain-of-thought reasoning
            max_new_tokens: Maximum number of new tokens to generate
            temperature: Temperature for generation
            top_p: Top-p sampling parameter
            do_sample: Whether to use sampling
            
        Returns:
            Generated reasoning text
        """
        # Process image if provided
        images = None
        if image is not None:
            processed_image = self.process_image(image)
            images = [processed_image.to(device=self.device, dtype=self.torch_dtype)]
        
        # Format prompt
        conv = copy.deepcopy(self.conv_template)
        
        # Add chain-of-thought instruction if requested
        cot_instruction = ""
        if use_cot:
            cot_instruction = (
                "Think through this step-by-step, showing your reasoning clearly. "
                "After your reasoning, provide your final answer."
            )
        
        # Add memory context if available
        memory_context = ""
        if memory_buffer and memory_buffer.entries:
            memory_context = "\n\n" + memory_buffer.to_prompt_format()
        
        # Format the user prompt
        if image is not None:
            user_prompt = f"{DEFAULT_IMAGE_TOKEN}\n{question}{memory_context}\n\n{cot_instruction}"
        else:
            user_prompt = f"{question}{memory_context}\n\n{cot_instruction}"
        
        conv.append_message(conv.roles[0], user_prompt)
        conv.append_message(conv.roles[1], "")
        prompt = conv.get_prompt()
        
        # Tokenize input
        inputs = self.vita_tokenizer([prompt], return_tensors="pt").to(self.device)
        
        # Generate response
        with torch.no_grad():
            outputs = self.vita_model.generate(
                **inputs,
                images=images,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=do_sample
            )
            
        output_text = self.vita_tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract just the model's response
        response = output_text.split(conv.roles[1] + ":")[-1].strip()
        
        return response
    
    def extract_final_answer(self, reasoning_text: str) -> str:
        """
        Extract the final answer from the reasoning text.
        
        Args:
            reasoning_text: Full reasoning text
            
        Returns:
            Extracted final answer
        """
        # Look for common answer indicators
        indicators = [
            "final answer:", 
            "final answer is:", 
            "therefore, the answer is:", 
            "the final solution is:",
            "the answer is:"
        ]
        
        lower_text = reasoning_text.lower()
        
        for indicator in indicators:
            if indicator in lower_text:
                index = lower_text.rfind(indicator)
                answer = reasoning_text[index + len(indicator):].strip()
                
                # If there are multiple sentences, take the first one
                if "." in answer:
                    sentences = answer.split(".")
                    answer = sentences[0].strip() + "."
                    
                return answer
        
        # If no indicator found, return the last sentence
        sentences = reasoning_text.split(".")
        if sentences:
            return sentences[-2].strip() + "." if len(sentences) > 1 else sentences[-1].strip() + "."
        
        return reasoning_text
    
    def evaluate_reasoning(
        self, 
        question: str, 
        reasoning_text: str,
        target_answer: Optional[str] = None
    ) -> Tuple[str, bool]:
        """
        Evaluate reasoning and generate feedback.
        
        Args:
            question: Original question
            reasoning_text: Generated reasoning text
            target_answer: Optional target answer
            
        Returns:
            Tuple of (feedback, is_correct)
        """
        # Extract final answer from reasoning
        final_answer = self.extract_final_answer(reasoning_text)
        
        # Use VITA to provide self-feedback
        conv = copy.deepcopy(self.conv_template)
        
        # Format the feedback prompt
        feedback_prompt = (
            f"Question: {question}\n\n"
            f"Reasoning: {reasoning_text}\n\n"
            f"Final Answer: {final_answer}\n\n"
        )
        
        if target_answer:
            feedback_prompt += f"Correct Answer: {target_answer}\n\n"
            
        feedback_prompt += (
            "Please evaluate the reasoning and identify any errors or improvements. "
            "Focus on logical errors, calculation mistakes, or misunderstandings of the problem. "
            "Be specific about what went wrong and how it should be corrected."
        )
        
        conv.append_message(conv.roles[0], feedback_prompt)
        conv.append_message(conv.roles[1], "")
        prompt = conv.get_prompt()
        
        # Tokenize input
        inputs = self.vita_tokenizer([prompt], return_tensors="pt").to(self.device)
        
        # Generate feedback
        with torch.no_grad():
            outputs = self.vita_model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.3,
                top_p=0.9,
                do_sample=True
            )
            
        output_text = self.vita_tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract just the model's response
        feedback = output_text.split(conv.roles[1] + ":")[-1].strip()
        
        # Determine if the reasoning is correct
        is_correct = False
        lower_feedback = feedback.lower()
        correct_indicators = [
            "correct",
            "accurate",
            "right",
            "appropriate",
            "valid",
            "accurate answer",
            "proper solution",
            "flawless"
        ]
        
        # Check if feedback suggests the reasoning is correct
        if any(indicator in lower_feedback for indicator in correct_indicators):
            negative_modifiers = [
                "not ", 
                "isn't ", 
                "doesn't ", 
                "wouldn't ", 
                "couldn't ", 
                "no ", 
                "error", 
                "mistake", 
                "incorrect"
            ]
            
            # Check if the correct indicators are negated
            is_correct = not any(
                modifier in lower_feedback[max(0, lower_feedback.find(indicator) - 10):lower_feedback.find(indicator)]
                for indicator in correct_indicators
                for modifier in negative_modifiers
                if indicator in lower_feedback
            )
        
        return feedback, is_correct
    
    def solve_with_memory(
        self, 
        question: str, 
        image: Optional[Union[str, Image.Image]] = None,
        target_answer: Optional[str] = None,
        max_iterations: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Solve a problem with iterative memory-augmented reasoning.
        
        Args:
            question: Question or problem to solve
            image: Optional image for the problem
            target_answer: Optional target answer for evaluation
            max_iterations: Maximum number of reasoning iterations
            
        Returns:
            Dictionary with results (iterations, final answer, correct)
        """
        if max_iterations is None:
            max_iterations = self.max_iterations
            
        # Initialize memory buffer
        self.rmm.clear_memory()
        memory_buffer = self.rmm.get_memory_buffer()
        
        iterations = []
        is_correct = False
        
        for i in range(max_iterations):
            # Generate reasoning
            self.logger.info(f"Generating reasoning (iteration {i+1}/{max_iterations})")
            reasoning_text = self.generate_reasoning(
                question=question, 
                image=image,
                memory_buffer=memory_buffer,
                use_cot=True
            )
            
            # Evaluate reasoning
            self.logger.info(f"Evaluating reasoning")
            feedback, is_correct = self.evaluate_reasoning(
                question=question,
                reasoning_text=reasoning_text,
                target_answer=target_answer
            )
            
            # Extract final answer
            final_answer = self.extract_final_answer(reasoning_text)
            
            # Store iteration results
            iteration_result = {
                "iteration": i+1,
                "reasoning": reasoning_text,
                "feedback": feedback,
                "memory_buffer": copy.deepcopy(memory_buffer),
                "is_correct": is_correct,
                "final_answer": final_answer
            }
            iterations.append(iteration_result)
            
            # If correct, no need for more iterations
            if is_correct:
                self.logger.info(f"Correct answer found after {i+1} iterations")
                break
                
            # Update memory buffer
            self.logger.info(f"Updating memory buffer")
            memory_buffer = self.rmm.update_memory(
                question=question,
                reasoning_steps=reasoning_text,
                feedback=feedback
            )
        
        # Return results
        result = {
            "question": question,
            "iterations": iterations,
            "final_answer": iterations[-1]["final_answer"],
            "is_correct": is_correct,
            "memory_buffer": memory_buffer,
            "num_iterations": len(iterations)
        }
        
        return result