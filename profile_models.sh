#!/bin/bash

# Check if dtype, seq_length, and master_port arguments are provided
if [ "$#" -lt 3 ]; then
    echo "Usage: $0 <dtype> <seq_length> <master_port> [model_names...]"
    exit 1
fi

dtype="$1"
seq_length="$2"
master_port="$3"
shift 3

# Define the array of default model names
default_model_names=("deepspeedgpt_1.3b" "deepspeedgpt_2.7b" "deepspeedgpt_6.7b" "deepspeedbert_large" "deepspeedbert_xlarge" "deepspeedvit_g" "deepspeedvit_e" "deepspeedllama_tiny" "deepspeedllama_3b" "deepspeedllama_7b")

# Function to profile a model
profile_model() {
  model_name="$1"
  echo "Profiling $model_name"
  python -m torch.distributed.run --nproc_per_node=1 --nnodes=1 --node_rank=0 --master_addr=localhost --master_port="$master_port" -m bench.bench_layer --seq_length "$seq_length" --profile_batches 4 --image_size 384 --trace_dir /tmp/pt_log --autocast_dtype "$dtype" --reduce_dtype "$dtype" --model_name "$model_name"
}

# Check if any model names are specified as command-line arguments
if [ "$#" -gt 0 ]; then
  # Loop through each specified model name and profile it
  for model_name in "$@"
  do
    profile_model "$model_name"
  done
else
  # If no models are specified, use the default list
  for model_name in "${default_model_names[@]}"
  do
    profile_model "$model_name"
  done
fi
