#!/bin/bash
echo "Running TALLRec training on GCP L4 GPU - GPU: $1, Seed: $2"
seed=$2

# L4 GPU Optimized Configuration
output_dir="/home/$(whoami)/TALLRec/output/movie"
base_model="/home/$(whoami)/TALLRec/models/llama-7b"  # or mistral-7b
train_data="/home/$(whoami)/TALLRec/data/movie/train.json"
val_data="/home/$(whoami)/TALLRec/data/movie/valid.json"
instruction_model="/home/$(whoami)/TALLRec/models/alpaca-lora-7b"

# Verify model paths exist
if [ ! -d "$base_model" ]; then
    echo "Error: Base model not found at $base_model"
    echo "Please download LLaMA-7B or Mistral-7B model first"
    exit 1
fi

if [ ! -d "$instruction_model" ]; then
    echo "Error: Instruction model not found at $instruction_model"
    echo "Please download Alpaca-LoRA weights first"
    exit 1
fi

# L4 has 24GB VRAM - optimize batch sizes accordingly
for lr in 1e-4
do
    for dropout in 0.05
    do
        for sample in 16 64 256  # Test all sample sizes from paper
        do
                mkdir -p $output_dir
                echo "lr: $lr, dropout: $dropout , seed: $seed, sample: $sample"
                
                # L4 optimized settings: larger batch size for better utilization
                CUDA_VISIBLE_DEVICES=$1 python -u finetune_rec.py \
                    --base_model $base_model \
                    --train_data_path $train_data \
                    --val_data_path $val_data \
                    --output_dir ${output_dir}_${seed}_${sample} \
                    --batch_size 256 \
                    --micro_batch_size 64 \
                    --num_epochs 200 \
                    --learning_rate $lr \
                    --cutoff_len 512 \
                    --lora_r 8 \
                    --lora_alpha 16\
                    --lora_dropout $dropout \
                    --lora_target_modules '[q_proj,v_proj]' \
                    --train_on_inputs \
                    --group_by_length \
                    --resume_from_checkpoint $instruction_model \
                    --sample $sample \
                    --seed $2 \
                    --fp16
        done
    done
done