import re
import numpy as np
from typing import Dict, List, Any, Tuple, Optional

class EvaluationMetrics:
    """Metrics for evaluating VITA with RMM performance"""
    
    @staticmethod
    def calculate_accuracy(predictions: List[Dict[str, Any]], targets: List[str]) -> float:
        """Calculate the final answer accuracy"""
        correct = 0
        for pred, target in zip(predictions, targets):
            if pred.get("is_correct", False):
                correct += 1
        return correct / len(predictions) if predictions else 0
    
    @staticmethod
    def calculate_step_correctness(predictions: List[Dict[str, Any]]) -> float:
        """Calculate the correctness of intermediate reasoning steps"""
        correct_steps = 0
        total_steps = 0
        
        for pred in predictions:
            iterations = pred.get("iterations", [])
            for iteration in iterations:
                correct_steps += iteration.get("step_correctness", 0)
                total_steps += iteration.get("total_steps", 0)
        
        return correct_steps / total_steps if total_steps else 0
    
    @staticmethod
    def calculate_error_repetition_rate(predictions: List[Dict[str, Any]]) -> float:
        """Measure how well the system avoids repeating errors after feedback"""
        repeated_errors = 0
        total_errors = 0
        
        for pred in predictions:
            iterations = pred.get("iterations", [])
            if len(iterations) < 2:
                continue
                
            for i in range(1, len(iterations)):
                prev_errors = iterations[i-1].get("errors", [])
                curr_errors = iterations[i].get("errors", [])
                
                for error in prev_errors:
                    if error in curr_errors:
                        repeated_errors += 1
                
                total_errors += len(prev_errors)
        
        return repeated_errors / total_errors if total_errors else 0
    
    @staticmethod
    def calculate_correction_effectiveness(predictions: List[Dict[str, Any]]) -> float:
        """Calculate the success rate of feedback in correcting errors"""
        initially_incorrect = 0
        later_correct = 0
        
        for pred in predictions:
            iterations = pred.get("iterations", [])
            if len(iterations) < 2:
                continue
                
            if not iterations[0].get("is_correct", False):
                initially_incorrect += 1
                if pred.get("is_correct", False):
                    later_correct += 1
        
        return later_correct / initially_incorrect if initially_incorrect else 0
    
    @staticmethod
    def evaluate_numerical_answer(prediction: str, target: str) -> bool:
        """Check if a numerical answer matches the target"""
        # Extract numbers from strings
        pred_numbers = re.findall(r"[-+]?\d*\.\d+|\d+", prediction)
        target_numbers = re.findall(r"[-+]?\d*\.\d+|\d+", target)
        
        if not pred_numbers or not target_numbers:
            return False
            
        # Compare the last number in each (usually the final answer)
        try:
            pred_value = float(pred_numbers[-1])
            target_value = float(target_numbers[-1])
            return abs(pred_value - target_value) < 1e-6
        except:
            return False
    
    @staticmethod
    def evaluate_multiple_choice(prediction: str, target: str) -> bool:
        """Check if a multiple choice answer matches the target"""
        # Extract option letter (A, B, C, D)
        pred_match = re.search(r'\b([A-D])[.\):]', prediction)
        target_match = re.search(r'\b([A-D])[.\):]', target)
        
        if not pred_match or not target_match:
            return False
            
        return pred_match.group(1) == target_match.group(1)