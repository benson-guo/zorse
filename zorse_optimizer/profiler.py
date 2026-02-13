import argparse
import csv
import json
import os
import subprocess
import tempfile
import time
from typing import Dict, List, Optional, Any, Tuple


def read_cluster_info(cluster_info_file: str) -> Dict[str, Any]:
    with open(cluster_info_file, "r") as f:
        return json.load(f)


def read_machine_file(
    machine_file: str,
) -> Tuple[
    List[str],
    Dict[str, str],
    Dict[str, str],
    Dict[str, str],
    Dict[str, str],
    Dict[str, str],
]:
    machines = []
    ssh_key_paths = {}
    conda_paths = {}
    conda_envs = {}
    repo_paths = {}
    nccl_socket_ifnames = {}

    with open(machine_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            parts = [part.strip() for part in line.split(",")]
            hostname = parts[0]
            if not hostname:
                continue

            machines.append(hostname)

            # Get SSH key path (optional)
            if len(parts) > 1 and parts[1]:
                ssh_key_paths[hostname] = parts[1]

            # Get conda path (required)
            if len(parts) > 2 and parts[2]:
                conda_paths[hostname] = parts[2]
            else:
                raise ValueError(f"Conda path not specified for machine {hostname}")

            # Get conda environment (required)
            if len(parts) > 3 and parts[3]:
                conda_envs[hostname] = parts[3]
            else:
                raise ValueError(
                    f"Conda environment not specified for machine {hostname}"
                )

            # Get repository path (required)
            if len(parts) > 4 and parts[4]:
                repo_paths[hostname] = parts[4]
            else:
                raise ValueError(
                    f"Repository path not specified for machine {hostname}"
                )

            # Get NCCL socket interface name (optional)
            if len(parts) > 5 and parts[5]:
                nccl_socket_ifnames[hostname] = parts[5]

    return (
        machines,
        ssh_key_paths,
        conda_paths,
        conda_envs,
        repo_paths,
        nccl_socket_ifnames,
    )


def profile_agrs_partition(
    *,
    group_info: Dict[str, Any],
    cluster_info: Dict[str, Any],
    model_name: str,
    autocast_dtype: str,
    reduce_dtype: str,
    machine_file: str,
    num_trials: int = 5,
    num_warmup: int = 5,
    nccl_ib_disable: bool = False,
    output_dir: str = "./results",
):
    """
    Profile all-gather/reduce-scatter operations on a group of GPUs.

    Args:
        group_info: Dictionary containing group configuration
        cluster_info: Dictionary containing cluster information
        model_name: Name of the model to profile
        autocast_dtype: Data type for autocasting
        reduce_dtype: Data type for reduction operations
        machine_file: Path to machine configuration file
        num_trials: Number of profiling trials
        num_warmup: Number of warmup iterations
        nccl_ib_disable: Whether to disable InfiniBand
        output_dir: Directory to store results

    Returns:
        Dictionary containing profiling results
    """
    # Read machine-specific configuration
    _, ssh_key_paths, conda_paths, conda_envs, repo_paths, nccl_socket_ifnames = (
        read_machine_file(machine_file)
    )

    group_id = group_info["group_id"]
    gpus = group_info["gpus"]
    machines = group_info["machines"]

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Create temporary result files
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as temp_file:
        local_result_file = temp_file.name

    timestamp = int(time.time())
    remote_result_file = f"/tmp/profile_result_{group_id}_{timestamp}.csv"

    # Multi-node setup
    if len(machines) > 1:
        master_machine = machines[0]
        processes = {}

        for machine in machines:
            ssh_key_path = ssh_key_paths.get(machine)
            if not ssh_key_path:
                print(
                    f"Warning: No SSH key path specified for {machine}, using default SSH config"
                )

            conda_path = conda_paths.get(machine)
            conda_env = conda_envs.get(machine)
            repo_path = repo_paths.get(machine)
            nccl_socket_ifname = nccl_socket_ifnames.get(machine, "")

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

            # Convert list of GPU indices to comma-separated string
            cuda_visible_devices = ",".join(str(gpu) for gpu in machine_gpus)

            # Set node rank based on machine position in the list
            node_rank = machines.index(machine)

            # Build the command
            cmd_parts = [
                f"cd {repo_path}",
                f"source {conda_path}",
                f"conda activate {conda_env}",
            ]

            # Add NCCL environment variables
            env_vars = []
            if nccl_socket_ifname:
                env_vars.append(f"NCCL_SOCKET_IFNAME={nccl_socket_ifname}")
            if nccl_ib_disable:
                env_vars.append("NCCL_IB_DISABLE=1")
            env_vars.append("CUDA_VISIBLE_DEVICES=" + cuda_visible_devices)

            # Join env vars and add torchrun command
            cmd_parts.append(
                " ".join(
                    env_vars
                    + [
                        "torchrun",
                        f"--nproc_per_node={len(machine_gpus)}",
                        f"--nnodes={len(machines)}",
                        f"--node_rank={node_rank}",
                        f"--master_addr={master_machine}",
                        f"--master_port=29500",
                        "run_comm_profiling.py",
                        f"--model_name {model_name}",
                        f"--autocast_dtype {autocast_dtype}",
                        f"--reduce_dtype {reduce_dtype}",
                        f"--num_trials {num_trials}",
                        f"--num_warmup {num_warmup}",
                        f"--output_file {remote_result_file}",
                    ]
                )
            )

            # Join all command parts
            cmd = " && ".join(cmd_parts)

            print(f"Running command on {machine}: {cmd}")
            # Use SSH key if provided, otherwise use default SSH config
            ssh_cmd = ["ssh"]
            if ssh_key_path:
                ssh_cmd.extend(["-i", ssh_key_path])
            ssh_cmd.extend([machine, cmd])

            process = subprocess.Popen(
                ssh_cmd,
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
        master_ssh_key = ssh_key_paths.get(master_machine)
        copy_cmd = "scp "
        if master_ssh_key:
            copy_cmd += f"-i {master_ssh_key} "
        copy_cmd += f"{master_machine}:{remote_result_file} {local_result_file}"
        print(f"Copying results from {master_machine}: {copy_cmd}")

        copy_process = subprocess.run(
            copy_cmd, shell=True, capture_output=True, text=True
        )
    else:
        # Single-node setup
        machine = machines[0]
        ssh_key_path = ssh_key_paths.get(machine)
        conda_path = conda_paths.get(machine)
        conda_env = conda_envs.get(machine)
        repo_path = repo_paths.get(machine)
        nccl_socket_ifname = nccl_socket_ifnames.get(machine, "")

        cuda_visible_devices = ",".join(str(gpu_local_idx) for gpu_local_idx, _ in gpus)

        # Build the command
        cmd_parts = [
            f"cd {repo_path}",
            f"source {conda_path}",
            f"conda activate {conda_env}",
        ]

        # Add NCCL environment variables
        env_vars = []
        if nccl_socket_ifname:
            env_vars.append(f"NCCL_SOCKET_IFNAME={nccl_socket_ifname}")
        if nccl_ib_disable:
            env_vars.append("NCCL_IB_DISABLE=1")
        env_vars.append("CUDA_VISIBLE_DEVICES=" + cuda_visible_devices)

        # Join env vars and add torchrun command
        cmd_parts.append(
            " ".join(
                env_vars
                + [
                    "torchrun",
                    f"--nproc_per_node={len(gpus)}",
                    f"--nnodes=1",
                    f"--node_rank=0",
                    f"--master_addr={machine}",
                    f"--master_port=29500",
                    "run_comm_profiling.py",
                    f"--model_name {model_name}",
                    f"--autocast_dtype {autocast_dtype}",
                    f"--reduce_dtype {reduce_dtype}",
                    f"--num_trials {num_trials}",
                    f"--num_warmup {num_warmup}",
                    f"--output_file {remote_result_file}",
                ]
            )
        )

        # Join all command parts
        cmd = " && ".join(cmd_parts)

        print(f"Running command on {machine}: {cmd}")

        ssh_cmd = ["ssh"]
        if ssh_key_path:
            ssh_cmd.extend(["-i", ssh_key_path])
        ssh_cmd.extend([machine, cmd])

        process = subprocess.run(ssh_cmd, capture_output=True, text=True)

        print(f"== Output from {machine} ==")
        print(process.stdout)
        if process.stderr:
            print(f"== Errors from {machine} ==")
            print(process.stderr)

        if process.returncode != 0:
            print(f"Command failed on {machine}")
            return None

        # Copy the result file from the remote machine to the local machine
        copy_cmd = "scp "
        if ssh_key_path:
            copy_cmd += f"-i {ssh_key_path} "
        copy_cmd += f"{machine}:{remote_result_file} {local_result_file}"
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
                break  # Only process the first row
        print(f"Successfully read results: {results}")
    except (FileNotFoundError, csv.Error) as e:
        print(f"Error reading results file: {e}")
        results = None

    # Clean up local temporary file
    try:
        os.remove(local_result_file)
    except OSError as e:
        print(f"Error removing temporary file: {e}")

    # Clean up remote temporary file on all machines
    for machine in machines:
        ssh_key_path = ssh_key_paths.get(machine)
        cleanup_cmd = "ssh "
        if ssh_key_path:
            cleanup_cmd += f"-i {ssh_key_path} "
        cleanup_cmd += f"{machine} 'rm -f {remote_result_file}'"
        subprocess.run(cleanup_cmd, shell=True, capture_output=True)

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Profile distributed all-gather and reduce-scatter operations"
    )
    parser.add_argument(
        "--cluster_info", type=str, required=True, help="Path to cluster info JSON file"
    )
    parser.add_argument(
        "--machine_file",
        type=str,
        required=True,
        help="Path to machine configuration file",
    )
    parser.add_argument(
        "--group_id", type=str, required=True, help="Group ID for the current run"
    )
    parser.add_argument(
        "--gpu_indices",
        type=str,
        required=True,
        help="Comma-separated list of GPU global ranks to use (e.g., '0,1,2,3')",
    )
    parser.add_argument(
        "--model_name", type=str, required=True, help="Name of model to profile"
    )
    parser.add_argument(
        "--autocast_dtype",
        type=str,
        default="float32",
        help="Data type for autocasting (default: float16)",
    )
    parser.add_argument(
        "--reduce_dtype",
        type=str,
        default="float32",
        help="Data type for reduction operations (default: float16)",
    )
    parser.add_argument(
        "--num_trials", type=int, default=5, help="Number of profiling trials"
    )
    parser.add_argument(
        "--num_warmup", type=int, default=5, help="Number of warmup iterations"
    )
    parser.add_argument(
        "--nccl_ib_disable", action="store_true", help="Disable InfiniBand for NCCL"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./results",
        help="Directory to store profiling results",
    )
    args = parser.parse_args()

    # Read cluster info
    cluster_info = read_cluster_info(args.cluster_info)

    # Parse GPU indices
    gpu_indices = [
        int(idx.strip()) for idx in args.gpu_indices.split(",") if idx.strip()
    ]

    # Get machines for these GPUs from the cluster info
    machines = set()
    gpus = []

    for gpu_idx in gpu_indices:
        for machine_info in cluster_info["machines"]:
            for gpu in machine_info["gpus"]:
                if gpu["global_rank"] == gpu_idx:
                    machines.add(machine_info["machine"])
                    gpus.append((gpu["index"], gpu_idx))  # (local_idx, global_idx)

    if not machines:
        print("Error: No machines found for the specified GPU indices")
        return

    # Create group info
    group_info = {"group_id": args.group_id, "gpus": gpus, "machines": list(machines)}

    # Run profiling
    results = profile_agrs_partition(
        group_info=group_info,
        cluster_info=cluster_info,
        model_name=args.model_name,
        autocast_dtype=args.autocast_dtype,
        reduce_dtype=args.reduce_dtype,
        machine_file=args.machine_file,
        num_trials=args.num_trials,
        num_warmup=args.num_warmup,
        nccl_ib_disable=args.nccl_ib_disable,
        output_dir=args.output_dir,
    )

    if results:
        # Save results to a file
        output_file = os.path.join(
            args.output_dir, f"profile_results_{args.group_id}.json"
        )
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {output_file}")
    else:
        print("Profiling failed or returned no results")


if __name__ == "__main__":
    main()
