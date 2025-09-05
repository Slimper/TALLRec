# TALLRec GCP Setup Guide

## Prerequisites for GCP Instance

1. **GCP Instance Requirements:**
   - Instance type: `g2-standard-4` or higher (4+ vCPUs, 16+ GB RAM) 
   - GPU: NVIDIA L4 (1+ GPUs recommended - perfect for LLaMA-7B)
   - Disk: 100+ GB SSD
   - OS: Ubuntu 22.04 LTS with Deep Learning VM image

2. **Required Model Files:**
   - LLaMA-7B model weights (Hugging Face format)
   - Instruction tuning model (e.g., Alpaca-LoRA weights)

## Setup Steps on GCP Instance

### 1. Setup GCP Instance with L4 GPU
```bash
# Create GCP instance (run this from your local machine)
gcloud compute instances create tallrec-l4 \
  --zone=us-central1-a \
  --machine-type=g2-standard-4 \
  --accelerator=type=nvidia-l4,count=1 \
  --image-family=deep-learning-platform-release \
  --image-project=ml-images \
  --boot-disk-size=100GB \
  --maintenance-policy=TERMINATE

# SSH into the instance
gcloud compute ssh tallrec-l4 --zone=us-central1-a

# Verify GPU (should show L4)
nvidia-smi
```

### 2. Install Python Environment
```bash
# Clone repository (if not already done)
git clone https://github.com/SAI990323/TALLRec.git
cd TALLRec

# The Deep Learning VM already has conda, use it for better GPU support
conda create -n tallrec python=3.10 -y
conda activate tallrec

# Install PyTorch with CUDA support (L4 supports CUDA 12.1)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install other dependencies  
pip install transformers peft datasets fire accelerate loralib bitsandbytes sentencepiece gradio

# Verify CUDA works
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0))"
```

### 3. Download Required Models

#### Option A: LLaMA-7B (Official Meta weights)
```bash
# Request access from Meta and download using Hugging Face
# You need to be approved for LLaMA access
huggingface-cli login
huggingface-cli download meta-llama/Llama-2-7b-hf --local-dir ./models/llama-7b
```

#### Option B: Alternative open models (if LLaMA unavailable)
```bash
# Use Mistral-7B or similar as alternative
huggingface-cli download mistralai/Mistral-7B-v0.1 --local-dir ./models/mistral-7b
```

#### Download Alpaca-LoRA weights
```bash
huggingface-cli download tloen/alpaca-lora-7b --local-dir ./models/alpaca-lora-7b
```

### 4. Verify GPU Setup
```bash
nvidia-smi
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('GPU count:', torch.cuda.device_count())"
```

## Running Experiments

The shell scripts need to be configured with the correct paths. See `shell/instruct_7B_gcp.sh` and `shell/evaluate_gcp.sh` for GCP-configured versions.

### Training
```bash
bash ./shell/instruct_7B_gcp.sh 0 42  # GPU 0, seed 42
```

### Evaluation  
```bash
bash ./shell/evaluate_gcp.sh 0 ./output/movie_42_64  # GPU 0, output directory
```

## Expected Results

The paper reports these AUC scores:
- Movie dataset: 67.24% (16-shot), 67.48% (64-shot), 71.98% (256-shot)
- Book dataset: 56.36% (16-shot), 60.39% (64-shot), 64.38% (256-shot)

Training typically takes 1-2 hours per experiment on an L4 GPU (faster than T4).