# VITA Reasoning Memory Manager (RMM)

This repository contains the implementation of a Reasoning Memory Manager (RMM) for the VITA multimodal large language model. The RMM augments VITA with a small, fine-tuned Qwen2.5-based model to handle memory management, enabling iterative reasoning with feedback.

## Overview

The VITA Reasoning Memory Manager extends VITA's reasoning capabilities by adding an iterative memory feedback loop. The system can:

1. Generate initial reasoning for problems (text-only or multimodal)
2. Evaluate the reasoning and identify errors
3. Create structured memory entries based on feedback
4. Incorporate memory entries in subsequent reasoning attempts
5. Iteratively improve answers through multiple reasoning cycles

## Architecture

The system consists of several key components:

- **VITA Base Model**: The foundation multimodal model (VITA-1.5)
- **Reasoning Memory Manager (RMM)**: A Qwen2.5-based model fine-tuned to manage memory entries
- **Memory Schema**: A structured format for storing and retrieving memory entries
- **Integration Layer**: Components for connecting VITA and RMM
- **Evaluation Pipeline**: Tools for measuring performance across benchmarks

## Installation

### Requirements

Below are the recommended requirements:
- Python 3.10+
- PyTorch 2.0+
- Transformers 4.30+
- PEFT (Parameter-Efficient Fine-Tuning) library
- Datasets
- PIL (Python Imaging Library)

### Setup

1. Unzip the repository and navigate to the directory:
```bash
cd vita-rmm/vita/rmm
```

2. Install the required packages:
```bash
conda create -n vitarmm python=3.10 -y
conda activate vitarmm
pip install --upgrade pip
pip install -r requirements.txt
```

3. Download pre-trained models and datasets from the google drive link https://drive.google.com/drive/folders/1mnd5DQZgwp5OafgwhhqyK0TWJkEOJe9u?usp=drive_link, and place them in the appropriate directories, then use the directories in the following commands. The downloaded files should include the following:
VITA-1.5 model weights, Qwen2.5 model weights, and LoRA adapter weights; GSM8K, MathVista, MMLU (With Vision Enchanced) dataset.

## Usage

### Data Generation and Simulation

To generate training data by simulating the reasoning-feedback loop:

```bash
python vita/rmm/data/simulation.py \
    --vita_model /path/to/vita_model \
    --feedback_model gpt-4 \  # or path to local feedback model
    --output_dir ./simulation_results \
    --gsm8k_path ./data/gsm8k/test.json \
    --mathvista_path ./data/mathvista/test.jsonl \
    --mmlu_path ./data/mmlu_vision/test.jsonl \
    --num_samples 100  # Set to -1 for all samples
```

### Training the Reasoning Memory Manager

To train the RMM model using the simulated data:

```bash
python vita/rmm/training/run_training.py \
    --model_name Qwen/Qwen2.5-7B \
    --output_dir ./rmm_output \
    --data_path ./simulation_results/combined_data.json \
    --lora_r 16 \
    --lora_alpha 32 \
    --lora_dropout 0.05 \
    --train_batch_size 4 \
    --eval_batch_size 4 \
    --num_epochs 3 \
    --gradient_accumulation_steps 8 \
    --learning_rate 2e-4
```

### Inference and Evaluation

To evaluate the full system on benchmark datasets:

```bash
python vita/rmm/evaluation/run_evaluation.py \
    --vita_model /path/to/vita_model \
    --vita_model_type mixtral-8x7b \  # Options: mixtral-8x7b, nemo, qwen2p5_instruct
    --rmm_base_model Qwen/Qwen2.5-7B \
    --rmm_lora_adapter ./rmm_output \
    --output_dir ./evaluation_results \
    --gsm8k_path ./data/gsm8k/test.json \
    --mathvista_path ./data/mathvista/test.jsonl \
    --mathvista_image_dir ./data/mathvista/images \
    --mmlu_path ./data/mmlu_vision/test.jsonl \
    --mmlu_image_dir ./data/mmlu_vision/images \
    --sample_size 100 \  # Set to -1 for all samples
    --max_iterations 3
```

## Model Components

### Memory Schema

The memory system uses a structured format for storing feedback and corrections:

- **MemoryEntry**: Individual correction or insight
  - `id`: Unique identifier
  - `entry_type`: Type of entry (correction, insight, fact)
  - `content`: Text content of the memory
  - `confidence`: Confidence score (0.0-1.0)
  - `related_to`: IDs of related entries
  - `metadata`: Additional information

- **MemoryBuffer**: Container for multiple memory entries
  - Methods for adding, retrieving, and formatting entries

### Reasoning Memory Manager

The RMM is a Qwen2.5-based model fine-tuned with LoRA to:

1. Process reasoning steps and feedback
2. Generate structured memory entries
3. Format memory for inclusion in future reasoning

### Integration with VITA

The system integrates RMM with VITA by:

1. Running initial reasoning with VITA
2. Generating feedback on the reasoning
3. Using RMM to create memory entries
4. Incorporating memory in VITA's context for subsequent attempts

## Benchmarks

The system is evaluated on three benchmarks:

1. **GSM8K**: Grade-school math problems (text-only)
2. **MathVista**: Visual math problems (multimodal)
3. **MMLU Vision**: Vision-augmented subset of MMLU (multimodal)

Key metrics include:

- **Final Answer Accuracy**: Percentage of correct answers
- **Step Correctness**: Correctness of intermediate reasoning steps
- **Error Repetition Rate**: How well the system avoids repeating errors
- **Correction Effectiveness**: Success rate of feedback in correcting errors

## Example

Here's an example of how the system works on a math problem:

1. **Initial Problem**:
   ```
   A square has a perimeter of 20 cm. What is its area in square centimeters?
   ```

2. **First Reasoning Attempt**:
   ```
   To find the area of a square, I need to find the side length first.
   The perimeter is 20 cm, which means the sum of all 4 sides is 20 cm.
   So each side is 20 cm / 4 = 5 cm.
   The area of a square is side length squared.
   Area = 5 cm × 5 cm = 25 square centimeters.
   ```

3. **Feedback**:
   ```
   The reasoning is correct. The perimeter of a square is the sum of all 4 sides, which is 20 cm.
   Each side length is 20 cm / 4 = 5 cm.
   The area of a square is side length squared, which is 5 cm × 5 cm = 25 square centimeters.
   ```

4. **Memory Entry**:
   ```
   [FACT] For a square with perimeter P, each side length is P/4, and the area is (P/4)².
   ```

5. **Second Problem**:
   ```
   A square has a perimeter of 36 cm. What is its area?
   ```

6. **Reasoning with Memory**:
   ```
   I remember that for a square with perimeter P, each side length is P/4, and the area is (P/4)².
   In this case, the perimeter P = 36 cm.
   So the side length = 36 cm / 4 = 9 cm.
   The area = 9 cm × 9 cm = 81 square centimeters.
   ```
