# Direct Preference Optimization for HTML Slide Generation

A machine learning project implementing Direct Preference Optimization (DPO) to fine-tune a language model for generating HTML presentation slides with human-preferred styling.
The project uses a synthetic dataset containing presentation slide prompts and two different slide designs for each prompt - one with a "rejected" style and one with a "chosen" style.
 
 ### What is Direct Preference Optimization (DPO)?

DPO is a powerful fine-tuning method that enables language models to learn from human preferences without the complexity of reinforcement learning. Unlike traditional approaches that require a reward model, DPO directly optimizes a model to prefer certain outputs over others based on paired examples (chosen vs. rejected completions). This approach is more computationally efficient and often produces better results than previous methods like RLHF (Reinforcement Learning from Human Feedback).

The key insight of DPO is that it reformulates the preference optimization problem into a classification problem: given a prompt and two completions, the model learns to assign higher probability to the preferred completion over the less preferred one.


## 🎯 Project Overview

This project demonstrates the application of preference learning techniques to visual design tasks, specifically fine-tuning a language model to generate clean, professional HTML presentation slides while avoiding overly decorative or flashy designs. The implementation achieves **100% preference accuracy** in distinguishing between preferred and non-preferred slide designs. 



## 🛠 Technical Stack

- **Framework**: PyTorch, Hugging Face Transformers
- **Fine-tuning**: PEFT (Parameter Efficient Fine-Tuning), TRL (Transformer Reinforcement Learning)
- **Optimization**: QLoRA (Quantized LoRA), BitsAndBytes for 4-bit quantization
- **Model**: Qwen2-1.5B-Instruct (1.5 billion parameters)
- **Data Processing**: BeautifulSoup, Datasets library
- **Visualization**: Matplotlib, Seaborn, Pandas

## 📊 Dataset and Problem Formulation

### Dataset Structure
The dataset is structured as a preference dataset, where each entry contains:
{
  "prompt": "Slide content description",
  "completion_a": "HTML for rejected-style slide",
  "completion_b": "HTML for chosen-style slide",
  "style_a": "rejected",
  "style_b": "chosen"
}
- **Size**: 100 carefully curated examples
- **Format**: JSON with prompt, chosen, and rejected completions
- **Split**: 80 training / 20 validation examples
- **Domain**: HTML presentation slides (16:9 aspect ratio, static CSS)


### Preference Criteria
The dataset distinguishes between:
- **Chosen (Preferred)**: Clean, professional, business-appropriate designs
- **Rejected (Non-preferred)**: Overly decorative, flashy, or cluttered designs


## 📁 Project Structure

```
DPO/
├── DPO_code.ipynb          # Complete implementation notebook
├── preference_dataset.json   # Training data (100 examples)
├── README.md               # This file
└── requirements.txt        # Project dependencies
```

## 🏃‍♂️ Quick Start

### Prerequisites
- Python 3.8 or higher
- **NVIDIA GPU with CUDA** (T4/L4/A10/A100, 16 GB + VRAM) **OR** access to a managed GPU runtime such as **Google Colab** or AWS EC2. Training will _not_ run on a CPU-only machine.
- Git



### 1. Set Up Virtual Environment

**Option A: Using venv (recommended)**
```bash
# Create virtual environment
python -m venv dpo_env

# Activate virtual environment
# On Windows:
dpo_env\Scripts\activate
# On macOS/Linux:
source dpo_env/bin/activate
```

### 2. Install Dependencies
```bash
# Install all required packages
pip install -r requirements.txt
```

> **Tip** – If you are using **Google Colab**, simply run the first notebook cell that installs the same `requirements.txt` and then continue. Colab provides a free T4/L4 GPU runtime that meets the project’s requirements.

### 3. Run the Project Locally (GPU workstation)
```bash
# Start Jupyter notebook
jupyter notebook DPO_slide.ipynb

# Or use Jupyter Lab
jupyter lab DPO_slide.ipynb
```

### 4. Run the Project on Google Colab
1. Open [`DPO_slide.ipynb`](DPO_slide.ipynb) on GitHub and click **“Open in Colab”** (Chrome extension) or upload the notebook directly to Colab.
2. Select **Runtime ▸ Change runtime type ▸ GPU (T4/L4/A100)**.
3. Execute the cells in order – the setup cell will install all dependencies automatically.
4. **Upload `preference_dataset.json`** (or mount it from Google Drive) into the Colab working directory before running the data-loading cell.

### 5. Notebook Sections Overview
- **Section 1**: Library installation and setup
- **Section 2**: Model loading and LoRA configuration
- **Section 3**: Dataset processing and preparation
- **Section 4**: DPO training configuration and execution
- **Section 5**: Evaluation and visualization
- **Section 6**: Base model comparison




## 🔧 Technical Implementation

### Model Architecture and Configuration

**Base Model Selection**: Qwen2-1.5B-Instruct
- Chosen for optimal balance between capability and resource constraints
- Instruction-tuned variant provides better following of HTML generation prompts
- 1.5B parameters manageable within T4 GPU memory limits

**QLoRA Configuration**:
```python
BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True
)
```
- **Rationale**: 4-bit quantization reduces memory usage by ~75% while maintaining performance
- **NF4 quantization**: Optimized for neural network weights distribution
- **Double quantization**: Additional memory savings for quantization constants

**LoRA Configuration**:
```python
LoraConfig(
    r=16,                    # Rank: Balance between efficiency and expressiveness
    lora_alpha=32,           # Scaling factor: 2x rank for stable training
    lora_dropout=0.1,        # Regularization to prevent overfitting
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]  # Attention projections
)
```
- **Target Modules**: Focus on attention mechanisms for better sequence understanding
- **Rank 16**: Sufficient capacity for style preference learning without overfitting


### Training Configuration

**DPO Hyperparameters**:
```python
DPOConfig(
    num_train_epochs=3,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,    # Effective batch size: 8
    learning_rate=1e-4,
    lr_scheduler_type="cosine",
    max_prompt_length=128,
    max_length=1200
)
```

**Design Rationale**:
- **3 Epochs**: Small dataset size requires careful balance to avoid overfitting
- **Gradient Accumulation**: Simulates larger batch sizes within memory constraints
- **Learning Rate 1e-4**: Conservative rate for stable preference learning
- **Cosine Scheduler**: Smooth learning rate decay for better convergence
- **Token Limits**: 128 for prompts (sufficient for slide descriptions), 1200 total (accommodates full HTML)



## 📈 Results and Performance

### Training Metrics Evolution

| Step | Validation Loss | Preference Accuracy | Reward Margin | Chosen LogP | Rejected LogP |
|------|----------------|-------------------|---------------|-------------|---------------|
| 10   | 0.0156         | 1.0000           | 5.84          | -386.17     | -557.08       |
| 20   | 0.0021         | 1.0000           | 9.56          | -400.96     | -609.11       |
| 30   | 0.0017         | 1.0000           | 9.97          | -402.86     | -615.04       |

### Key Observations

1. **Immediate Perfect Accuracy**: Model achieved 100% preference accuracy by step 10
2. **Consistent Improvement**: Validation loss continued decreasing throughout training
3. **Increasing Confidence**: Growing reward margins indicate stronger preference differentiation
4. **Stable Convergence**: Log probabilities stabilized without oscillation

### Performance Analysis

**Why Such Rapid Success?**
- **Model Capacity**: 1.5B parameters provide sufficient representational power compared to very small dataset (20 validation entries)
- **Clear Task Definition**: HTML styling preferences are visually distinct
- **Quality Dataset**: Well-curated examples with clear preference signals
- **Effective Architecture**: Attention mechanisms naturally suited for sequence-to-sequence tasks

## 🔍 Evaluation Framework

### Comprehensive Metrics Dashboard

The evaluation system tracks six key metrics:

1. **Validation Loss**: Overall model performance
2. **Reward Scores**: Separate tracking for chosen vs rejected completions
3. **Preference Accuracy**: Binary classification performance
4. **Reward Margins**: Confidence in preference decisions
5. **Log Probabilities**: Model confidence in generated sequences
6. **Logits Analysis**: Raw model outputs for deeper insights

### Base Model Comparison

Implemented evaluation pipeline for comparing fine-tuned model against base Qwen2-1.5B-Instruct:
- Processes same validation examples
- Computes preference accuracy and margins
- Visualizes performance differences
- Quantifies improvement from DPO training

## 💡 Technical Challenges and Solutions

### Memory Management
**Challenge**: T4 GPU memory limitations (16GB) with 1.5B parameter model
**Solution**: 
- QLoRA 4-bit quantization (~75% memory reduction)
- Gradient accumulation (effective larger batches)
- Explicit memory clearing between training phases
