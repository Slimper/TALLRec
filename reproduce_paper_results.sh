#!/bin/bash
# Reproduce TALLRec paper results
# Usage: bash reproduce_paper_results.sh [gpu_id] [random_seed]

GPU_ID=${1:-0}
SEED=${2:-42}

echo "========================================="
echo "Reproducing TALLRec Paper Results"
echo "GPU: $GPU_ID, Seed: $SEED"
echo "========================================="

# Check if models are available
BASE_MODEL="/home/$(whoami)/TALLRec/models/llama-7b"
INSTRUCTION_MODEL="/home/$(whoami)/TALLRec/models/alpaca-lora-7b"

if [ ! -d "$BASE_MODEL" ]; then
    echo "ERROR: Base model not found at $BASE_MODEL"
    echo "Please download LLaMA-7B model first using:"
    echo "  huggingface-cli download meta-llama/Llama-2-7b-hf --local-dir ./models/llama-7b"
    echo "Or download Mistral-7B as alternative:"
    echo "  huggingface-cli download mistralai/Mistral-7B-v0.1 --local-dir ./models/mistral-7b"
    exit 1
fi

if [ ! -d "$INSTRUCTION_MODEL" ]; then
    echo "ERROR: Instruction model not found at $INSTRUCTION_MODEL"
    echo "Please download Alpaca-LoRA weights first using:"
    echo "  huggingface-cli download tloen/alpaca-lora-7b --local-dir ./models/alpaca-lora-7b"
    exit 1
fi

echo "Models found. Starting experiments..."
echo ""

# Part 1: Movie Dataset Experiments
echo "========================================="
echo "MOVIE DATASET EXPERIMENTS"
echo "========================================="

echo "Training TALLRec on movie dataset..."
bash ./shell/instruct_7B_gcp.sh $GPU_ID $SEED

echo "Evaluating movie results..."
bash ./shell/evaluate_gcp.sh $GPU_ID ./output/movie

# Part 2: Book Dataset Experiments  
echo ""
echo "========================================="
echo "BOOK DATASET EXPERIMENTS"
echo "========================================="

echo "Training TALLRec on book dataset..."
bash ./shell/book_experiment_gcp.sh $GPU_ID $SEED

echo "Evaluating book results..."
bash ./shell/evaluate_book_gcp.sh $GPU_ID ./output/book

echo ""
echo "========================================="
echo "EXPERIMENTS COMPLETE"
echo "========================================="

echo "Results saved to:"
echo "  Movie results: ./output/movie.json" 
echo "  Book results: ./output/book.json"
echo ""

echo "Expected results from paper:"
echo "Movie dataset AUC:"
echo "  16-shot: 67.24%"
echo "  64-shot: 67.48%" 
echo "  256-shot: 71.98%"
echo ""
echo "Book dataset AUC:"
echo "  16-shot: 56.36%"
echo "  64-shot: 60.39%"
echo "  256-shot: 64.38%"
echo ""

echo "Check the JSON files above to compare your results with the paper."