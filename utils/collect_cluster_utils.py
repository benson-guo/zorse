import argparse
import subprocess

from utils.comm import clean_gpu_name


def read_machine_file(machine_file: str):
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

            # Get SSH key path for this machine (optional)
            if len(parts) > 1 and parts[1]:
                ssh_key_paths[hostname] = parts[1]

            # Get conda path for this machine (required)
            if len(parts) > 2 and parts[2]:
                conda_paths[hostname] = parts[2]
            else:
                raise ValueError(f"Conda path not specified for machine {hostname}")

            # Get conda env for this machine (required)
            if len(parts) > 3 and parts[3]:
                conda_envs[hostname] = parts[3]
            else:
                raise ValueError(
                    f"Conda environment not specified for machine {hostname}"
                )

            # Get repository path for this machine (required)
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


def get_gpu_info(machine: str, ssh_key_path: str = None):
    command = "nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader"
    ssh_cmd = ["ssh"]

    if ssh_key_path:
        ssh_cmd.extend(["-i", ssh_key_path])

    ssh_cmd.extend([machine, command])

    try:
        result = subprocess.run(
            ssh_cmd,
            capture_output=True,
            text=True,
            check=True,
        )
    except subprocess.CalledProcessError as e:
        print(f"Error fetching GPU info from {machine}: {e.stderr}")
        return []

    gpus = []
    for line in result.stdout.strip().split("\n"):
        if not line.strip():
            continue
        parts = [x.strip() for x in line.split(",")]
        if len(parts) == 3:
            gpu_index, gpu_name, gpu_memory = parts
            try:
                mem_val = int(gpu_memory.split()[0])
            except ValueError:
                mem_val = 0
            gpus.append(
                {
                    "index": int(gpu_index),
                    "name": clean_gpu_name(gpu_name),
                    "memory_total_mib": mem_val,
                }
            )
    return gpus


def collect_cluster_info(machines, ssh_key_paths):
    """Collect information about all GPUs in the cluster."""
    cluster_info = {"machines": [], "total_gpus": 0, "gpu_list": []}

    gpu_id_counter = 0
    global_rank_map = {}

    for machine_idx, machine in enumerate(machines):
        ssh_key_path = ssh_key_paths.get(machine)
        gpu_list = get_gpu_info(machine, ssh_key_path)
        machine_gpus = []

        for gpu in gpu_list:
            global_rank = gpu_id_counter
            gpu_with_id = {
                **gpu,
                "id": f"gpu_{gpu_id_counter}",
                "machine": machine,
                "global_rank": global_rank,
            }
            machine_gpus.append(gpu_with_id)
            cluster_info["gpu_list"].append(gpu_with_id)

            global_rank_map[global_rank] = {
                "machine": machine,
                "local_rank": gpu["index"],
                "machine_index": machine_idx,
            }

            gpu_id_counter += 1

        machine_info = {
            "machine": machine,
            "num_gpus": len(gpu_list),
            "gpus": machine_gpus,
            "machine_index": machine_idx,
        }
        cluster_info["machines"].append(machine_info)
        cluster_info["total_gpus"] += len(gpu_list)

    cluster_info["global_rank_map"] = global_rank_map
    return cluster_info

def parse_args():
    parser = argparse.ArgumentParser(
        description="Collect cluster GPU information and measure interconnect bandwidth."
    )
    parser.add_argument(
        "--machine_file",
        type=str,
        required=True,
        help="File containing list of machines in the cluster",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="cluster_info.json",
        help="Output JSON file for the cluster information",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./results",
        help="Directory to store benchmark results",
    )
    parser.add_argument(
        "--ib_disable",
        action="store_true",
        help="Disable InfiniBand for the bandwidth test",
    )
    parser.add_argument(
        "--skip_internode", action="store_true", help="Skip internode bandwidth testing"
    )
    parser.add_argument(
        "--skip_intranode", action="store_true", help="Skip intranode bandwidth testing"
    )
    parser.add_argument(
        "--skip_bandwidth", action="store_true", help="Skip all bandwidth testing"
    )
    parser.add_argument(
        "--parallel",
        type=int,
        default=1,
        help="Number of parallel communications for bandwidth testing",
    )
    parser.add_argument(
        "--send_msg_size",
        type=int,
        default=10**7,
        help="Size of send/recv message for bandwidth testing",
    )
    parser.add_argument(
        "--optimize_internode",
        action="store_true",
        help="Optimize internode tests by grouping similar machines",
    )

    args = parser.parse_args()
    return args