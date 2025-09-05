#!/bin/bash
# Download required models for TALLRec
echo "========================================="
echo "Downloading Models for TALLRec"
echo "========================================="

# Create models directory
mkdir -p models
cd models

echo "You have two options for the base model:"
echo "1. LLaMA-7B (requires Meta approval)"
echo "2. Mistral-7B (open access)"
echo ""

read -p "Which model would you like to download? (1 for LLaMA, 2 for Mistral): " choice

if [ "$choice" = "1" ]; then
    echo "Downloading LLaMA-7B model..."
    echo "Note: You need to be approved for LLaMA access and logged into Hugging Face CLI"
    echo "If you haven't logged in yet, run: huggingface-cli login"
    echo ""
    
    huggingface-cli download meta-llama/Llama-2-7b-hf --local-dir ./llama-7b
    
    if [ $? -eq 0 ]; then
        echo "✓ LLaMA-7B downloaded successfully"
    else
        echo "✗ Failed to download LLaMA-7B. Make sure you have access and are logged in."
        exit 1
    fi
    
elif [ "$choice" = "2" ]; then
    echo "Downloading Mistral-7B model..."
    
    huggingface-cli download mistralai/Mistral-7B-v0.1 --local-dir ./mistral-7b
    
    if [ $? -eq 0 ]; then
        echo "✓ Mistral-7B downloaded successfully"
        echo "Note: You'll need to update the shell scripts to use mistral-7b instead of llama-7b"
    else
        echo "✗ Failed to download Mistral-7B"
        exit 1
    fi
    
else
    echo "Invalid choice. Exiting."
    exit 1
fi

echo ""
echo "Downloading Alpaca-LoRA instruction tuning model..."

huggingface-cli download tloen/alpaca-lora-7b --local-dir ./alpaca-lora-7b

if [ $? -eq 0 ]; then
    echo "✓ Alpaca-LoRA downloaded successfully"
else
    echo "✗ Failed to download Alpaca-LoRA"
    exit 1
fi

echo ""
echo "========================================="
echo "Download Complete!"
echo "========================================="
echo "Downloaded models:"
ls -la

echo ""
echo "You can now run the experiments using:"
echo "  bash reproduce_paper_results.sh 0 42"
echo ""
echo "Or run individual experiments:"
echo "  bash ./shell/instruct_7B_gcp.sh 0 42"
echo "  bash ./shell/evaluate_gcp.sh 0 ./output/movie"