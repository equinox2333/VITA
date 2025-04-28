import os
import json
import random
import torch
import logging
from typing import List, Dict, Any, Tuple, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import argparse
import numpy as np
from PIL import Image

from vita.model.vita_arch import VITAMetaForCausalLM
from vita.constants import DEFAULT_IMAGE_TOKEN
from vita.util.mm_utils import process_images

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class FeedbackSimulator:
    def __init__(
        self,
        vita_model_path: str,
        feedback_model_path: str,
        output_dir: str,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.device = device
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        logger.info(f"Loading VITA model from {vita_model_path}")
        self.vita_model_type = self._determine_model_type(vita_model_path)
        self.vita_model, self.vita_tokenizer, self.vita_image_processor = self._load_vita_model(vita_model_path)
        
        logger.info(f"Loading feedback model from {feedback_model_path}")
        self.feedback_model, self.feedback_tokenizer = self._load_feedback_model(feedback_model_path)
        
        # Templates for prompting
        self.cot_template = "Please solve this problem step-by-step using chain-of-thought reasoning. Write out each step clearly and give your final answer at the end."
        self.feedback_template = """
You are an expert evaluator helping to improve reasoning. Review the following problem and the model's solution:

Problem: {problem}

Model's Solution:
{solution}

Give detailed and helpful feedback on any errors or improvements in the reasoning. If there are errors, explain what went wrong and suggest corrections.
Focus on:
1. Mathematical errors
2. Logical fallacies
3. Misinterpretations of the problem or images
4. Missing important steps or concepts

Your Feedback:
"""
        self.memory_template = """
Based on the feedback provided, create a structured memory entry that captures key corrections for future reasoning. Format it as:

ERROR_TYPE: [math_error/logic_error/misinterpretation/missing_step]
CORRECTION: [concise statement of the correction]
CONTEXT: [specific part of reasoning where this applies]
"""

    def _determine_model_type(self, model_path: str) -> str:
        if "mixtral" in model_path.lower():
            return "mixtral-8x7b"
        elif "nemo" in model_path.lower():
            return "nemo"
        elif "qwen" in model_path.lower():
            return "qwen2p5_instruct"
        else:
            return "mixtral-8x7b"  # Default

    def _load_vita_model(self, model_path: str):
        from vita.model.builder import load_pretrained_model
        
        tokenizer, model, image_processor, _ = load_pretrained_model(
            model_path=model_path,
            model_base=None,
            model_name="vita-1.5",
            model_type=self.vita_model_type,
            load_8bit=False,
            load_4bit=True,
            device_map="auto",
        )
        return model, tokenizer, image_processor

    def _load_feedback_model(self, model_path: str):
        # Load feedback model (GPT-4 API or local model)
        if model_path.startswith("gpt-"):
            # Using OpenAI API
            try:
                import openai
                return model_path, None
            except ImportError:
                raise ImportError("OpenAI package is required to use GPT models. Install with: pip install openai")
        else:
            # Local model
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True,
            )
            tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=True,
                use_fast=True
            )
            return model, tokenizer

    def _process_image(self, image_path: str):
        try:
            image = Image.open(image_path).convert('RGB')
            processed_image = process_images([image], self.vita_image_processor, self.vita_model.config)
            return processed_image
        except Exception as e:
            logger.error(f"Error processing image {image_path}: {e}")
            return None

    def generate_vita_reasoning(self, problem: Dict[str, Any]) -> str:
        try:
            # Prepare input for VITA
            if "image_path" in problem and problem["image_path"]:
                # Process image
                image = self._process_image(problem["image_path"])
                if image is None:
                    raise ValueError(f"Failed to process image at {problem['image_path']}")
                
                # Create prompt with image token
                prompt = f"{DEFAULT_IMAGE_TOKEN}\n{problem['question']}\n\n{self.cot_template}"
                
                # Generate reasoning with VITA
                input_ids = self.vita_tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
                with torch.inference_mode():
                    outputs = self.vita_model.generate(
                        inputs=input_ids,
                        images=image.to(self.device),
                        max_new_tokens=512,
                        temperature=0.7,
                        top_p=0.9,
                        do_sample=True
                    )
                response = self.vita_tokenizer.decode(outputs[0], skip_special_tokens=True)
                
                # Extract reasoning part
                reasoning = response.split(self.cot_template)[-1].strip()
            else:
                # Text-only problem
                prompt = f"{problem['question']}\n\n{self.cot_template}"
                
                # Generate reasoning with VITA
                input_ids = self.vita_tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
                with torch.inference_mode():
                    outputs = self.vita_model.generate(
                        input_ids=input_ids,
                        max_new_tokens=512,
                        temperature=0.7,
                        top_p=0.9,
                        do_sample=True
                    )
                response = self.vita_tokenizer.decode(outputs[0], skip_special_tokens=True)
                
                # Extract reasoning part
                reasoning = response.split(self.cot_template)[-1].strip()
            
            return reasoning
        
        except Exception as e:
            logger.error(f"Error generating VITA reasoning: {e}")
            return ""

    def generate_feedback(self, problem: Dict[str, Any], solution: str) -> str:
        try:
            # Format problem for feedback
            problem_text = problem["question"]
            if "image_path" in problem and problem["image_path"]:
                problem_text = f"[This problem includes an image] {problem_text}"
            
            # Prepare feedback prompt
            feedback_prompt = self.feedback_template.format(
                problem=problem_text,
                solution=solution
            )
            
            # Generate feedback
            if isinstance(self.feedback_model, str) and self.feedback_model.startswith("gpt-"):
                # Using OpenAI API
                import openai
                response = openai.ChatCompletion.create(
                    model=self.feedback_model,
                    messages=[
                        {"role": "system", "content": "You are an expert evaluator helping improve reasoning."},
                        {"role": "user", "content": feedback_prompt}
                    ],
                    temperature=0.3,
                    max_tokens=1024
                )
                feedback = response.choices[0].message.content
            else:
                # Local model
                inputs = self.feedback_tokenizer(feedback_prompt, return_tensors="pt").to(self.device)
                with torch.inference_mode():
                    outputs = self.feedback_model.generate(
                        input_ids=inputs.input_ids,
                        attention_mask=inputs.attention_mask,
                        max_new_tokens=512,
                        temperature=0.3,
                        top_p=0.9
                    )
                feedback = self.feedback_tokenizer.decode(outputs[0], skip_special_tokens=True)
                feedback = feedback[len(feedback_prompt):].strip()
            
            return feedback
        
        except Exception as e:
            logger.error(f"Error generating feedback: {e}")
            return ""

    def generate_memory_entry(self, feedback: str) -> Dict[str, str]:
        try:
            # Prepare memory entry prompt
            memory_prompt = f"{feedback}\n\n{self.memory_template}"
            
            # Generate memory entry
            if isinstance(self.feedback_model, str) and self.feedback_model.startswith("gpt-"):
                # Using OpenAI API
                import openai
                response = openai.ChatCompletion.create(
                    model=self.feedback_model,
                    messages=[
                        {"role": "system", "content": "You are an expert at distilling feedback into structured memory entries."},
                        {"role": "user", "content": memory_prompt}
                    ],
                    temperature=0.3,
                    max_tokens=512
                )
                memory_text = response.choices[0].message.content
            else:
                # Local model
                inputs = self.feedback_tokenizer(memory_prompt, return_tensors="pt").to(self.device)
                with torch.inference_mode():
                    outputs = self.feedback_model.generate(
                        input_ids=inputs.input_ids,
                        attention_mask=inputs.attention_mask,
                        max_new_tokens=256,
                        temperature=0.3,
                        top_p=0.9
                    )
                memory_text = self.feedback_tokenizer.decode(outputs[0], skip_special_tokens=True)
                memory_text = memory_text[len(memory_prompt):].strip()
            
            # Parse memory entry
            memory_entry = {}
            lines = memory_text.strip().split('\n')
            for line in lines:
                if ":" in line:
                    key, value = line.split(":", 1)
                    memory_entry[key.strip()] = value.strip()
            
            # Ensure all required fields are present
            required_fields = ["ERROR_TYPE", "CORRECTION", "CONTEXT"]
            for field in required_fields:
                if field not in memory_entry:
                    memory_entry[field] = ""
            
            return memory_entry
        
        except Exception as e:
            logger.error(f"Error generating memory entry: {e}")
            return {"ERROR_TYPE": "", "CORRECTION": "", "CONTEXT": ""}

    def generate_second_attempt(self, problem: Dict[str, Any], initial_solution: str, memory_entry: Dict[str, str]) -> str:
        try:
            # Format memory for prompt
            memory_text = "\n".join([f"{k}: {v}" for k, v in memory_entry.items()])
            
            # Create prompt with memory feedback
            if "image_path" in problem and problem["image_path"]:
                # Process image
                image = self._process_image(problem["image_path"])
                if image is None:
                    raise ValueError(f"Failed to process image at {problem['image_path']}")
                
                prompt = f"{DEFAULT_IMAGE_TOKEN}\n{problem['question']}\n\nYour previous attempt: {initial_solution}\n\nFeedback on your reasoning:\n{memory_text}\n\nNow, solve the problem again step-by-step, correcting the errors mentioned in the feedback."
                
                # Generate second attempt with VITA
                input_ids = self.vita_tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
                with torch.inference_mode():
                    outputs = self.vita_model.generate(
                        inputs=input_ids,
                        images=image.to(self.device),
                        max_new_tokens=512,
                        temperature=0.7,
                        top_p=0.9,
                        do_sample=True
                    )
                response = self.vita_tokenizer.decode(outputs[0], skip_special_tokens=True)
                
                # Extract reasoning part
                second_attempt = response.split("Now, solve the problem again")[-1].strip()
            else:
                prompt = f"{problem['question']}\n\nYour previous attempt: {initial_solution}\n\nFeedback on your reasoning:\n{memory_text}\n\nNow, solve the problem again step-by-step, correcting the errors mentioned in the feedback."
                
                # Generate second attempt with VITA
                input_ids = self.vita_tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
                with torch.inference_mode():
                    outputs = self.vita_model.generate(
                        input_ids=input_ids,
                        max_new_tokens=512,
                        temperature=0.7,
                        top_p=0.9,
                        do_sample=True
                    )
                response = self.vita_tokenizer.decode(outputs[0], skip_special_tokens=True)
                
                # Extract reasoning part
                second_attempt = response.split("Now, solve the problem again")[-1].strip()
            
            return second_attempt
        
        except Exception as e:
            logger.error(f"Error generating second attempt: {e}")
            return ""

    def process_dataset(self, dataset_path: str, dataset_type: str, num_samples: int = -1) -> List[Dict[str, Any]]:
        logger.info(f"Processing {dataset_type} dataset from {dataset_path}")
        
        # Load dataset based on type
        problems = self._load_dataset(dataset_path, dataset_type)
        
        # Limit samples if specified
        if num_samples > 0:
            problems = problems[:num_samples]
        
        results = []
        for i, problem in enumerate(tqdm(problems, desc=f"Processing {dataset_type}")):
            try:
                # Generate initial reasoning with VITA
                initial_solution = self.generate_vita_reasoning(problem)
                if not initial_solution:
                    logger.warning(f"Failed to generate initial solution for problem {i}. Skipping.")
                    continue
                
                # Generate feedback
                feedback = self.generate_feedback(problem, initial_solution)
                if not feedback:
                    logger.warning(f"Failed to generate feedback for problem {i}. Skipping.")
                    continue
                
                # Generate memory entry
                memory_entry = self.generate_memory_entry(feedback)
                
                # Generate second attempt with feedback
                second_attempt = self.generate_second_attempt(problem, initial_solution, memory_entry)
                
                # Store results
                result = {
                    "problem_id": problem.get("id", f"{dataset_type}_{i}"),
                    "dataset": dataset_type,
                    "question": problem["question"],
                    "image_path": problem.get("image_path", ""),
                    "ground_truth": problem.get("answer", ""),
                    "initial_solution": initial_solution,
                    "feedback": feedback,
                    "memory_entry": memory_entry,
                    "second_attempt": second_attempt
                }
                
                results.append(result)
                
                # Save intermediate results
                if (i + 1) % 10 == 0:
                    self._save_results(results, f"{dataset_type}_intermediate_{i}.json")
            
            except Exception as e:
                logger.error(f"Error processing problem {i}: {e}")
                continue
        
        # Save final results
        self._save_results(results, f"{dataset_type}_final.json")
        
        return results

    def _load_dataset(self, dataset_path: str, dataset_type: str) -> List[Dict[str, Any]]:
        problems = []
        
        try:
            if dataset_type.lower() == "gsm8k":
                with open(dataset_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                for i, item in enumerate(data):
                    problems.append({
                        "id": f"gsm8k_{i}",
                        "question": item["question"],
                        "answer": item.get("answer", ""),
                        "image_path": ""  # GSM8K is text-only
                    })
            
            elif dataset_type.lower() == "mathvista":
                with open(dataset_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                for i, item in enumerate(data):
                    image_path = item.get("image_path", "")
                    # Fix relative paths if needed
                    if image_path and not os.path.isabs(image_path):
                        image_path = os.path.join(os.path.dirname(dataset_path), image_path)
                    
                    problems.append({
                        "id": item.get("id", f"mathvista_{i}"),
                        "question": item["question"],
                        "answer": item.get("answer", ""),
                        "image_path": image_path
                    })
            
            elif dataset_type.lower() == "mmlu":
                with open(dataset_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                for i, item in enumerate(data):
                    image_path = item.get("image_path", "")
                    # Fix relative paths if needed
                    if image_path and not os.path.isabs(image_path):
                        image_path = os.path.join(os.path.dirname(dataset_path), image_path)
                    
                    problems.append({
                        "id": item.get("id", f"mmlu_{i}"),
                        "question": item["question"],
                        "options": item.get("options", []),
                        "answer": item.get("answer", ""),
                        "image_path": image_path
                    })
            
            else:
                raise ValueError(f"Unsupported dataset type: {dataset_type}")
        
        except Exception as e:
            logger.error(f"Error loading dataset {dataset_path}: {e}")
        
        return problems

    def _save_results(self, results: List[Dict[str, Any]], filename: str):
        output_path = os.path.join(self.output_dir, filename)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved results to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Generate synthetic data for RMM training")
    parser.add_argument("--vita_model", type=str, required=True, help="Path to VITA model")
    parser.add_argument("--feedback_model", type=str, required=True, help="Path to feedback model or 'gpt-4'")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for simulation results")
    parser.add_argument("--gsm8k_path", type=str, default="", help="Path to GSM8K dataset")
    parser.add_argument("--mathvista_path", type=str, default="", help="Path to MathVista dataset")
    parser.add_argument("--mmlu_path", type=str, default="", help="Path to MMLU dataset")
    parser.add_argument("--num_samples", type=int, default=-1, help="Number of samples to process per dataset (-1 for all)")
    
    args = parser.parse_args()
    
    simulator = FeedbackSimulator(
        vita_model_path=args.vita_model,
        feedback_model_path=args.feedback_model,
        output_dir=args.output_dir
    )
    
    if args.gsm8k_path:
        simulator.process_dataset(args.gsm8k_path, "gsm8k", args.num_samples)
    
    if args.mathvista_path:
        simulator.process_dataset(args.mathvista_path, "mathvista", args.num_samples)
    
    if args.mmlu_path:
        simulator.process_dataset(args.mmlu_path, "mmlu", args.num_samples)

if __name__ == "__main__":
    main()