# Zorse

[![DOI](https://zenodo.org/badge/DOI/YOUR_DOI_HERE.svg)](https://doi.org/YOUR_DOI_HERE)

Zorse is a system for efficient LLM training on heterogeneous GPU clusters. It uses Pipeline-Efficient ZeRO DP, a novel integration of pipeline parallelism and data parallelism that is both communication- and memory-efficient.

## Setup
The setup script will create a conda environment and install Pytorch 2.5.1 as well as other libraries needed to run the code. Tested with CUDA 12.1, PyTorch 2.5.1.

```sh
./setup.sh
```

## Execution
The following is an example for training LLama 1B model with a batch size of 128 on a cluster of 4 GPUS: 2xL4, 1xA6000, 1xP40.

<h4>Profiling</h4>
First run the profiler to profile the model layer runtimes. This will profile every distinct GPU on the machine:

```sh
./profile_models.sh float16 <sequence_len> <port> deepspeedllama_tiny
```

Then, profile the cluster information/bandwidths.
```sh
python3 collect_cluster_info.py --machine_file hostfile --ib_disable --output example/cluster_info.json
```

The hostfile should specify the machines in the following format

```sh
#hostname,ssh_key_path,conda_path,conda_env,repo_path,NCCL_SOCKET_IFNAME
<machine_ip>,<optional_ssh_key>,<conda_path>,zorse,<zorse_dir>,<socket_name>
```
The profiled stats will be dumped to `cluster_info.json`

<h4>Planner</h4>
Then, run the optimizer with the profiled stats to generate an optimized training configuration.

```sh
python3 zorse_planner.py --cluster_info_file example/cluster_info.json --output example/training_config.json --global_batch_size 128 --sequence_length <sequence_length> --use_agrs_comm_model --model_name deepspeedllama_tiny --machine_file example/hostfile
```

<h4>Training</h4>
Then run the trainer with the configuration produced from the planner:


```sh
torchrun --nproc_per_node=4 --nnodes=1 --node_rank=0 --master_addr=localhost --master_port=12345 zorse.py --config_file example/training_config.json --zero2_pipeline --gloo_p2p --offload_model_params --optimizer_in_backwards
```
Note: May need to set environment variables `GLOO_SOCKET_IFNAME`, `NCCL_SOCKET_IFNAME`

<h4>Distributed Training</h4>
When running training on a cluster with multiple nodes, the memory profiling and training commands need to be run on all nodes in the cluster. You will need to add and configure the following flags to torchrun: --nnodes, --node_rank, --master_addr, and --master_port

## Expected Results

Each step of the pipeline produces the following output:

- **Profiling (`profile_models.sh`):** Per-layer forward/backward pass timings for each distinct GPU type, saved to a JSON profile under `profiled_models/`.
- **Cluster Info (`collect_cluster_info.py`):** A `cluster_info.json` file containing detected GPU types, memory capacities, and measured inter-GPU communication bandwidths.
- **Planner (`zorse_planner.py`):** A `training_config.json` file specifying the optimal parallelism strategy — pipeline stage assignments, data parallelism groups, and micro-batch schedules.
- **Training (`zorse.py`):** Per-iteration timing logs, memory usage statistics, and a final Training Summary with throughput (tokens/s), per-GPU memory utilization, time per step (ms), and Model FLOPs Utilization (MFU).

## Chameleon Cloud Artifact

A ready-to-run Trovi artifact is available for reproducing results on Chameleon Cloud:

**[Trovi Artifact](https://trovi.chameleoncloud.org/dashboard/artifacts/a93d14aa-4c2b-4748-9c5e-cbd28b663020)**

See [`chameleon.ipynb`](chameleon.ipynb) for the complete end-to-end walkthrough. The notebook was originally tested on 4x NVIDIA Tesla V100-PCIE-32GB.
