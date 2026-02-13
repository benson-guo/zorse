import csv
import os
import subprocess
import tempfile
import time
from typing import List, Dict
from grolar_optimizer.cluster import ModelConfig
from grolar_optimizer.constants import GPU_MEMORY_STATS


def validate_memory_for_stage_config(
    stages_config: List,
    model_config: ModelConfig,
    dtype_multiplier: float,
    optimizer_multiplier: float,
    microbatch_size: int,
    reserved_memory_gb: float,
) -> Dict[int, bool]:
    group_to_stages = {}
    for stage in stages_config:
        if stage.group.group_id not in group_to_stages:
            group_to_stages[stage.group.group_id] = []
        group_to_stages[stage.group.group_id].append(stage)

    def get_num_layers(stage_config):
        return stage_config.layer_end - stage_config.layer_start

    group_to_max_layers = {}
    for g_id, stages in group_to_stages.items():
        max_layers_needed = get_num_layers(stages[0])
        for i in range(len(stages) - 1):
            max_layers_needed = max(
                max_layers_needed,
                get_num_layers(stages[i]) + get_num_layers(stages[i + 1]),
            )
        group_to_max_layers[g_id] = max_layers_needed

    # Validate memory requirements for each group
    group_validation_results = {}

    for g_id, stages in group_to_stages.items():
        # Get GPU information for this group
        group = stages[0].group
        num_gpus_in_group = group.get_num_gpus()

        # Calculate total layers in this group
        layers_in_group = [get_num_layers(stage) for stage in stages]
        total_layers_in_group = sum(layers_in_group)
        max_layers_per_group = group_to_max_layers[g_id]

        # Find minimum GPU memory in the group
        gpu_memories = [GPU_MEMORY_STATS[gpu.identifier] for gpu in group.gpus]
        min_gpu_memory_gb = min(gpu_memories)

        # validate per group
        # model params - unsharded parameters that need to be stored in full
        model_params = (
            max_layers_per_group * model_config.params_per_layer * dtype_multiplier
        )

        # sharded params - parameters that can be sharded across GPUs
        sharded_model_params = (
            (total_layers_in_group - max_layers_per_group)
            * model_config.params_per_layer
            * dtype_multiplier
            / num_gpus_in_group
        )

        # activation memory
        activations_mem = (
            microbatch_size
            * model_config.sequence_length
            * model_config.hidden_size
            * dtype_multiplier
            * 2
        )

        # optimizer
        optimizer_mem = (
            total_layers_in_group
            * model_config.params_per_layer
            * optimizer_multiplier
            / num_gpus_in_group
        )

        # gpu activation and grad buffer
        pipeline_mem = (
            microbatch_size
            * model_config.sequence_length
            * model_config.hidden_size
            * dtype_multiplier
        )

        # Calculate total memory required in bytes
        total_bytes = (
            model_params
            + sharded_model_params
            + activations_mem
            + optimizer_mem
            + pipeline_mem
        )

        # Convert to GB and add reserved memory
        total_gb = (total_bytes / (1024**3)) + reserved_memory_gb

        # Validate if minimum GPU memory in the group is sufficient
        group_validation_results[g_id] = total_gb <= min_gpu_memory_gb

        print(
            f"For group {g_id} : Memory estimated {total_gb}, minimum GPU memory in group {min_gpu_memory_gb}"
        )
        print(
            f"\tmodel_params: {model_params}, sharded_model_params: {sharded_model_params}, activations_mem: {activations_mem}, optimizer_mem: {optimizer_mem}, pipeline_mem: {pipeline_mem}"
        )

    return group_validation_results


def profile_agrs_gpu_group(
    group_info,
    cluster_info,
    model_name,
    autocast_dtype,
    reduce_dtype,
    conda_env="grolar",
    repo_path="~/fall2023/groler",
    num_trials=5,
    num_warmup=5,
    is_paper=False,
    ssh_key_paths=None,
    conda_paths=None,
):

    # Default values if not provided
    if ssh_key_paths is None:
        ssh_key_paths = {
            m: "/home/ubuntu/efs/keys/ml-bguo.pem" for m in group_info["machines"]
        }
    if conda_paths is None:
        conda_paths = {
            m: "/opt/conda/etc/profile.d/conda.sh" for m in group_info["machines"]
        }

    group_id = group_info["group_id"]
    gpus = group_info["gpus"]
    machines = group_info["machines"]
    group_id = group_info["group_id"]
    gpus = group_info["gpus"]
    machines = group_info["machines"]
    # Create a unique temporary file on the local machine (where the script is running)
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as temp_file:
        local_result_file = temp_file.name

    # Create a temporary result file path for remote machines
    remote_result_file = f"/tmp/profile_result_{group_id}_{int(time.time())}.csv"

    # Multi-node setup
    if len(machines) > 1:
        master_machine = machines[0]
        processes = {}

        for machine in machines:
            ssh_key_path = ssh_key_paths.get(
                machine, "/home/ubuntu/efs/keys/ml-bguo.pem"
            )
            conda_path = conda_paths.get(machine, "/opt/conda/etc/profile.d/conda.sh")
            # Get the GPUs on this machine for this group
            machine_gpus = []
            for gpu_local_idx, gpu_global_idx in gpus:
                # Find GPUs that belong to this machine in the cluster info
                for gpu in cluster_info["gpu_list"]:
                    if (
                        gpu["machine"] == machine
                        and gpu["global_rank"] == gpu_global_idx
                    ):
                        machine_gpus.append(gpu_local_idx)

            if not machine_gpus:
                continue

            # Get the local indices of these GPUs
            cuda_visible_devices = ",".join(str(gpu) for gpu in machine_gpus)

            # Set node rank based on machine position in the list
            node_rank = machines.index(machine)

            # multi-node paper needs IB disable and socket name
            if is_paper:
                cmd = (
                    f"cd {repo_path} && "
                    f"source {conda_path} && "
                    f"conda activate {conda_env} && "
                    f'NCCL_IB_DISABLE=1 NCCL_SOCKET_IFNAME="enp1s0f0" '
                    f"CUDA_VISIBLE_DEVICES={cuda_visible_devices} "
                    f"torchrun "
                    f"--nproc_per_node={len(machine_gpus)} "
                    f"--nnodes={len(machines)} "
                    f"--node_rank={node_rank} "
                    f"--master_addr={master_machine} "
                    f"--master_port=29500 "
                    f"run_comm_profiling.py "
                    f"--model_name {model_name} "
                    f"--autocast_dtype {autocast_dtype} "
                    f"--reduce_dtype {reduce_dtype} "
                    f"--num_trials {num_trials} "
                    f"--num_warmup {num_warmup} "
                    f"--output_file {remote_result_file}"
                )
            else:
                cmd = (
                    f"cd {repo_path} && "
                    f"source {conda_path} && "
                    f"conda activate {conda_env} && "
                    f"CUDA_VISIBLE_DEVICES={cuda_visible_devices} "
                    f"torchrun "
                    f"--nproc_per_node={len(machine_gpus)} "
                    f"--nnodes={len(machines)} "
                    f"--node_rank={node_rank} "
                    f"--master_addr={master_machine} "
                    f"--master_port=29500 "
                    f"run_comm_profiling.py "
                    f"--model_name {model_name} "
                    f"--autocast_dtype {autocast_dtype} "
                    f"--reduce_dtype {reduce_dtype} "
                    f"--num_trials {num_trials} "
                    f"--num_warmup {num_warmup} "
                    f"--output_file {remote_result_file}"
                )

            print(f"Running command on {machine}: {cmd}")
            process = subprocess.Popen(
                ["ssh", "-i", ssh_key_path, machine, cmd],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )

            processes[machine] = process

        # Wait for all processes to complete
        for machine, process in processes.items():
            stdout, stderr = process.communicate()
            print(f"== Output from {machine} ==")
            print(stdout)
            if stderr:
                print(f"== Errors from {machine} ==")
                print(stderr)
            print(f"Process on {machine} exited with code {process.returncode}")
            if process.returncode != 0:
                print(f"Command failed on {machine}")
                return None

        # Copy the result file from the master machine to the local machine
        master_ssh_key = ssh_key_paths.get(
            master_machine, "/home/ubuntu/efs/keys/ml-bguo.pem"
        )
        copy_cmd = f"scp -i {master_ssh_key} {master_machine}:{remote_result_file} {local_result_file}"
        print(f"Copying results from {master_machine}: {copy_cmd}")

        copy_process = subprocess.run(
            copy_cmd, shell=True, capture_output=True, text=True
        )
    else:
        # Single-node setup
        machine = machines[0]
        ssh_key_path = ssh_key_paths.get(machine, "/home/ubuntu/efs/keys/ml-bguo.pem")
        conda_path = conda_paths.get(machine, "/opt/conda/etc/profile.d/conda.sh")
        cuda_visible_devices = ",".join(
            str(gpu_local_idx) for gpu_local_idx, gpu_global_idx in gpus
        )

        cmd = (
            f"cd {repo_path} && "
            f"source {conda_path} && "
            f"conda activate {conda_env} && "
            f"CUDA_VISIBLE_DEVICES={cuda_visible_devices} "
            f"torchrun "
            f"--nproc_per_node={len(gpus)} "
            f"--nnodes=1 "
            f"--node_rank=0 "
            f"--master_addr={machine} "
            f"--master_port=29500 "
            f"run_comm_profiling.py "
            f"--model_name {model_name} "
            f"--autocast_dtype {autocast_dtype} "
            f"--reduce_dtype {reduce_dtype} "
            f"--num_trials {num_trials} "
            f"--num_warmup {num_warmup} "
            f"--output_file {remote_result_file}"
        )

        print(f"Running command on {machine}: {cmd}")

        process = subprocess.run(
            ["ssh", "-i", ssh_key_path, machine, cmd], capture_output=True, text=True
        )

        print(f"== Output from {machine} ==")
        print(process.stdout)
        if process.stderr:
            print(f"== Errors from {machine} ==")
            print(process.stderr)

        if process.returncode != 0:
            print(f"Command failed on {machine}")
            return None

        # Copy the result file from the remote machine to the local machine
        copy_cmd = (
            f"scp -i {ssh_key_path} {machine}:{remote_result_file} {local_result_file}"
        )
        print(f"Copying results from {machine}: {copy_cmd}")

        copy_process = subprocess.run(
            copy_cmd, shell=True, capture_output=True, text=True
        )

        if copy_process.returncode != 0:
            print(f"Failed to copy results from {machine}: {copy_process.stderr}")
            return None

    # Parse the results from the local file
    results = {}
    try:
        print(f"Reading results from local file: {local_result_file}")
        with open(local_result_file, "r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                results = {
                    "group_id": group_id,
                    "model_name": row["model_name"],
                    "num_params": int(float(row["num_params"])),
                    "message_size_mb": float(row["message_size_mb"]),
                    "num_gpus": int(row["num_gpus"]),
                    "allgather": {
                        "avg_time": float(row["allgather_avg_time"]),
                        "min_time": float(row["allgather_min_time"]),
                        "max_time": float(row["allgather_max_time"]),
                        "std_dev": float(row["allgather_std_dev"]),
                        "bandwidth": float(row["allgather_bandwidth"]),
                    },
                    "reducescatter": {
                        "avg_time": float(row["reducescatter_avg_time"]),
                        "min_time": float(row["reducescatter_min_time"]),
                        "max_time": float(row["reducescatter_max_time"]),
                        "std_dev": float(row["reducescatter_std_dev"]),
                        "bandwidth": float(row["reducescatter_bandwidth"]),
                    },
                }
                break
        print(f"Successfully read results: {results}")
    except (FileNotFoundError, csv.Error) as e:
        print(f"Error reading results file: {e}")
        results = None

    # Clean up local temporary file
    try:
        os.remove(local_result_file)
    except OSError as e:
        print(f"Error removing temporary file: {e}")

    # Also clean up remote temporary file on all machines
    for machine in machines:
        cleanup_cmd = f"ssh -i /home/ubuntu/efs/keys/ml-bguo.pem {machine} 'rm -f {remote_result_file}'"
        subprocess.run(cleanup_cmd, shell=True, capture_output=True)

    return results
