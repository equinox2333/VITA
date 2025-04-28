import os
import json
import random
import pandas as pd
from typing import Dict, List, Any, Tuple, Optional, Callable
from tqdm import tqdm
from PIL import Image
import torch

from vita.rmm.inference.pipeline import VITAMemoryPipeline
from vita.rmm.evaluation.metrics import EvaluationMetrics

class BenchmarkRunner:
    """Runner for benchmark evaluation tasks"""
    
    def __init__(
        self, 
        pipeline: VITAMemoryPipeline,
        output_dir: str = "results",
        max_iterations: int = 3,
    ):
        """
        Initialize benchmark runner
        
        Args:
            pipeline: Initialized VITA with Memory pipeline
            output_dir: Directory to save results
            max_iterations: Maximum iterations for solving each problem
        """
        self.pipeline = pipeline
        self.output_dir = output_dir
        self.max_iterations = max_iterations
        self.metrics = EvaluationMetrics()
        
        os.makedirs(output_dir, exist_ok=True)
    
    def run_gsm8k(self, data_path: str, sample_size: Optional[int] = None) -> Dict[str, float]:
        """
        Run evaluation on GSM8K dataset
        
        Args:
            data_path: Path to GSM8K dataset
            sample_size: Number of samples to evaluate (None for all)
            
        Returns:
            Dictionary of metrics
        """
        print(f"Loading GSM8K dataset from {data_path}")
        
        with open(data_path, 'r') as f:
            dataset = json.load(f)
        
        if sample_size:
            if sample_size > len(dataset):
                print(f"Warning: Requested sample size {sample_size} is larger than dataset size {len(dataset)}")
                sample_size = len(dataset)
            dataset = random.sample(dataset, sample_size)
        
        results = []
        targets = []
        
        for item in tqdm(dataset, desc="Evaluating GSM8K"):
            question = item["question"]
            answer = item["answer"]
            
            result = self.pipeline.solve_with_memory(
                question=question,
                target_answer=answer,
                max_iterations=self.max_iterations
            )
            
            results.append(result)
            targets.append(answer)
        
        # Calculate metrics
        metrics = self._calculate_metrics(results, targets)
        
        # Save results
        self._save_results("gsm8k", results, metrics)
        
        return metrics
    
    def run_mathvista(self, data_path: str, image_dir: str, sample_size: Optional[int] = None) -> Dict[str, float]:
        """
        Run evaluation on MathVista dataset
        
        Args:
            data_path: Path to MathVista dataset
            image_dir: Directory containing images
            sample_size: Number of samples to evaluate (None for all)
            
        Returns:
            Dictionary of metrics
        """
        print(f"Loading MathVista dataset from {data_path}")
        
        # MathVista is typically in jsonl format
        dataset = []
        with open(data_path, 'r') as f:
            for line in f:
                dataset.append(json.loads(line))
        
        if sample_size:
            if sample_size > len(dataset):
                print(f"Warning: Requested sample size {sample_size} is larger than dataset size {len(dataset)}")
                sample_size = len(dataset)
            dataset = random.sample(dataset, sample_size)
        
        results = []
        targets = []
        
        for item in tqdm(dataset, desc="Evaluating MathVista"):
            question = item["problem"]
            answer = item["answer"]
            image_path = os.path.join(image_dir, item["image_filename"])
            
            # Load image
            try:
                image = Image.open(image_path).convert('RGB')
            except:
                print(f"Error loading image {image_path}, skipping")
                continue
            
            result = self.pipeline.solve_with_memory(
                question=question,
                image=image,
                target_answer=answer,
                max_iterations=self.max_iterations
            )
            
            results.append(result)
            targets.append(answer)
        
        # Calculate metrics
        metrics = self._calculate_metrics(results, targets)
        
        # Save results
        self._save_results("mathvista", results, metrics)
        
        return metrics
    
    def run_mmlu_vision(self, data_path: str, image_dir: str, sample_size: Optional[int] = None) -> Dict[str, float]:
        """
        Run evaluation on vision-augmented MMLU dataset
        
        Args:
            data_path: Path to MMLU dataset
            image_dir: Directory containing images
            sample_size: Number of samples to evaluate (None for all)
            
        Returns:
            Dictionary of metrics
        """
        print(f"Loading MMLU vision dataset from {data_path}")
        
        dataset = []
        with open(data_path, 'r') as f:
            for line in f:
                dataset.append(json.loads(line))
        
        if sample_size:
            if sample_size > len(dataset):
                print(f"Warning: Requested sample size {sample_size} is larger than dataset size {len(dataset)}")
                sample_size = len(dataset)
            dataset = random.sample(dataset, sample_size)
        
        results = []
        targets = []
        
        for item in tqdm(dataset, desc="Evaluating MMLU Vision"):
            question = item["question"]
            options = item["options"]
            answer = item["answer"]
            image_path = os.path.join(image_dir, item["image_filename"])
            
            # Format question with options
            formatted_question = question + "\n\n"
            for i, option in enumerate(options):
                formatted_question += f"{chr(65+i)}. {option}\n"
            
            # Load image
            try:
                image = Image.open(image_path).convert('RGB')
            except:
                print(f"Error loading image {image_path}, skipping")
                continue
            
            result = self.pipeline.solve_with_memory(
                question=formatted_question,
                image=image,
                target_answer=answer,
                max_iterations=self.max_iterations
            )
            
            results.append(result)
            targets.append(answer)
        
        # Calculate metrics
        metrics = self._calculate_metrics(results, targets)
        
        # Save results
        self._save_results("mmlu_vision", results, metrics)
        
        return metrics
    
    def run_all_benchmarks(
        self, 
        gsm8k_path: str, 
        mathvista_path: str, 
        mathvista_image_dir: str, 
        mmlu_path: str, 
        mmlu_image_dir: str,
        sample_size: Optional[int] = None
    ) -> Dict[str, Dict[str, float]]:
        """
        Run evaluation on all benchmarks
        
        Args:
            gsm8k_path: Path to GSM8K dataset
            mathvista_path: Path to MathVista dataset
            mathvista_image_dir: Directory containing MathVista images
            mmlu_path: Path to MMLU dataset
            mmlu_image_dir: Directory containing MMLU images
            sample_size: Number of samples to evaluate per benchmark (None for all)
            
        Returns:
            Dictionary of metrics for each benchmark
        """
        results = {}
        
        results["gsm8k"] = self.run_gsm8k(gsm8k_path, sample_size)
        results["mathvista"] = self.run_mathvista(mathvista_path, mathvista_image_dir, sample_size)
        results["mmlu_vision"] = self.run_mmlu_vision(mmlu_path, mmlu_image_dir, sample_size)
        
        # Save combined results
        combined_metrics = {
            "gsm8k": results["gsm8k"],
            "mathvista": results["mathvista"],
            "mmlu_vision": results["mmlu_vision"]
        }
        
        with open(os.path.join(self.output_dir, "all_benchmarks_results.json"), 'w') as f:
            json.dump(combined_metrics, f, indent=2)
        
        # Print summary table
        print("\n===== Benchmark Results =====")
        metrics_df = pd.DataFrame({
            'Benchmark': ["GSM8K", "MathVista", "MMLU Vision"],
            'Accuracy': [
                results["gsm8k"]["accuracy"] * 100,
                results["mathvista"]["accuracy"] * 100,
                results["mmlu_vision"]["accuracy"] * 100
            ],
            'Step Correctness (%)': [
                results["gsm8k"]["step_correctness"] * 100,
                results["mathvista"]["step_correctness"] * 100,
                results["mmlu_vision"]["step_correctness"] * 100
            ],
            'Error Repetition (%)': [
                results["gsm8k"]["error_repetition_rate"] * 100,
                results["mathvista"]["error_repetition_rate"] * 100,
                results["mmlu_vision"]["error_repetition_rate"] * 100
            ],
            'Correction Rate (%)': [
                results["gsm8k"]["correction_effectiveness"] * 100,
                results["mathvista"]["correction_effectiveness"] * 100,
                results["mmlu_vision"]["correction_effectiveness"] * 100
            ]
        })
        
        print(metrics_df.to_string(index=False))
        
        return combined_metrics
    
    def _calculate_metrics(self, results: List[Dict], targets: List[str]) -> Dict[str, float]:
        """Calculate all metrics for a batch of results"""
        metrics = {
            "accuracy": self.metrics.calculate_accuracy(results, targets),
            "step_correctness": self.metrics.calculate_step_correctness(results),
            "error_repetition_rate": self.metrics.calculate_error_repetition_rate(results),
            "correction_effectiveness": self.metrics.calculate_correction_effectiveness(results)
        }
        return metrics
    
    def _save_results(self, benchmark_name: str, results: List[Dict], metrics: Dict[str, float]) -> None:
        """Save results and metrics to disk"""
        output_file = os.path.join(self.output_dir, f"{benchmark_name}_results.json")
        with open(output_file, 'w') as f:
            json.dump({
                "metrics": metrics,
                "results": results
            }, f, indent=2)
        
        print(f"Results saved to {output_file}")
        print(f"Metrics for {benchmark_name}:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")