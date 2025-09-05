#!/bin/bash
echo "Running TALLRec book evaluation on GCP - GPU: $1, Output dir: $2"
CUDA_ID=$1
output_dir=$2
model_path=$(ls -d $output_dir*)

# GCP Configuration for book dataset
base_model="/home/$(whoami)/TALLRec/models/llama-7b"  # or mistral-7b
test_data="/home/$(whoami)/TALLRec/data/book/test.json"

# Verify paths
if [ ! -d "$base_model" ]; then
    echo "Error: Base model not found at $base_model"
    exit 1
fi

if [ ! -f "$test_data" ]; then
    echo "Error: Test data not found at $test_data"
    exit 1
fi

echo "Found model checkpoints:"
for path in $model_path
do
    echo "  $path"
done

echo ""
echo "Starting book evaluation..."

for path in $model_path
do
    echo "Evaluating: $path"
    CUDA_VISIBLE_DEVICES=$CUDA_ID python evaluate.py \
        --base_model $base_model \
        --lora_weights $path \
        --test_data_path $test_data \
        --result_json_data $2.json
done

echo "Book evaluation complete. Results saved to $2.json"