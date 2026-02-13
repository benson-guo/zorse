import re
import subprocess


def parse_intranode_bandwidth_from_output(output):
    # Look for "Bandwidth: X.XX Gb/s" or "Intranode Bandwidth (minimum of X pairs): X.XX Gb/s"
    bandwidth_patterns = [
        r"Intranode Bandwidth \(minimum of \d+ pairs\):\s+([\d.]+)\s+Gb/s",
        r"Bandwidth:\s+([\d.]+)\s+Gb/s",
    ]

    latency_patterns = [
        r"Intranode Latency \(minimum of \d+ pairs\):\s+([\d.]+)\s+ms",
        r"Latency:\s+([\d.]+)\s+ms",
    ]

    bandwidth = 0.0
    latency = 0.0

    # Parse bandwidth
    for pattern in bandwidth_patterns:
        matches = re.findall(pattern, output)
        if matches:
            try:
                bandwidth = float(matches[-1])
                break
            except (ValueError, IndexError):
                continue

    # Parse latency
    for pattern in latency_patterns:
        matches = re.findall(pattern, output)
        if matches:
            try:
                latency = float(matches[-1])
                break
            except (ValueError, IndexError):
                continue

    return bandwidth, latency


def get_gpu_ranks_by_machine(cluster_info):
    """
    Group GPU global ranks by machine for bandwidth testing.
    Returns a string formatted as: "rank1,rank2#rank3,rank4"
    where # separates machines and commas separate GPUs within a machine.
    """
    gpu_ranks_by_machine = []

    for machine_info in cluster_info["machines"]:
        machine_ranks = []
        for gpu in machine_info["gpus"]:
            machine_ranks.append(str(gpu["global_rank"]))

        gpu_ranks_by_machine.append(",".join(machine_ranks))

    return "#".join(gpu_ranks_by_machine)


def run_intranode_bandwidth_test(
    machine,
    machine_idx,
    num_gpus,
    conda_path,
    conda_env,
    repo_path,
    nccl_socket_ifname,
    ssh_key_path,
    output_dir,
    ib_disable=False,
    parallel=1,
    send_msg_size=10**7,
):
    cmd_parts = [
        f"cd {repo_path}",
        f"source {conda_path}",
        f"conda activate {conda_env}",
        "lsof -t -i :12355 > /dev/null 2>&1 && lsof -t -i :12355 | xargs kill -9 || echo 'No processes to kill on port 12355'",
        "lsof -t -i :12345 > /dev/null 2>&1 && lsof -t -i :12345 | xargs kill -9 || echo 'No processes to kill on port 12345'",
        "lsof -t -i :12346 > /dev/null 2>&1 && lsof -t -i :12346 | xargs kill -9 || echo 'No processes to kill on port 12346'",
    ]
    dist_run_parts = []

    # Add NCCL_SOCKET_IFNAME if specified
    if nccl_socket_ifname:
        dist_run_parts.append(f"NCCL_SOCKET_IFNAME={nccl_socket_ifname}")

    if ib_disable:
        dist_run_parts.append("NCCL_IB_DISABLE=1")

    # Extract just this machine's GPU ranks
    machine_ranks = ",".join([str(i) for i in range(num_gpus)])

    dist_run_parts.extend(
        [
            "python -m torch.distributed.run",
            f"--nproc_per_node={num_gpus}",
            f"--nnodes=1",
            f"--node_rank=0",
            f"--master_addr={machine}",  # Use the machine itself as master for intranode test
            f"--master_port=12346",
            "benchmark_gpu_bandwidth.py",
            f'--node_ranks "{machine_ranks}"',
            f"--parallel={parallel}",
            f"--send_msg_size={send_msg_size}",
            "--intranode",
            "--fast_intra",
            "--warmup_iterations 5",
            "--iterations 5",
            f"--output_file {output_dir}/intranode_bandwidth_results_{machine_idx}.json",
        ]
    )

    cmd_parts.append(" ".join(dist_run_parts))

    cmd_str = " && ".join(cmd_parts)

    print(f"Launching intranode bandwidth test on {machine}...")

    ssh_cmd = ["ssh"]

    if ssh_key_path:
        ssh_cmd.extend(["-i", ssh_key_path])
        print(f"Command: ssh -i {ssh_key_path} {machine} '{cmd_str}'")
    else:
        print(f"Command: ssh {machine} '{cmd_str}'")

    ssh_cmd.append(machine)
    ssh_cmd.append(cmd_str)

    process = subprocess.Popen(
        ssh_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    return process


def parse_internode_bandwidth_from_output(stdout_data):
    """Parse bandwidth directly from command output."""
    bandwidth_pattern = r"Bandwidth:\s+([\d.]+)\s+Gb/s"
    latency_pattern = r"Latency:\s+([\d.]+)\s+ms"

    bandwidth = 0.0
    latency = 0.0

    # Find all bandwidth measurements
    bandwidth_matches = re.findall(bandwidth_pattern, stdout_data)
    if bandwidth_matches:
        bandwidth_values = [float(match) for match in bandwidth_matches]
        bandwidth = sum(bandwidth_values) / len(bandwidth_values)

    # Find all latency measurements
    latency_matches = re.findall(latency_pattern, stdout_data)
    if latency_matches:
        latency_values = [float(match) for match in latency_matches]
        latency = sum(latency_values) / len(latency_values)

    return bandwidth, latency


def run_internode_bandwidth_test(
    src_machine,
    dst_machine,
    src_idx,
    dst_idx,
    src_machine_info,
    dst_machine_info,
    conda_path,
    conda_env,
    repo_path,
    nccl_socket_ifname,
    ssh_key_path,
    output_dir,
    ib_disable=False,
    parallel=1,
    send_msg_size=10**7,
):
    src_ranks = [str(gpu["global_rank"]) for gpu in src_machine_info["gpus"]]
    dst_ranks = [str(gpu["global_rank"]) for gpu in dst_machine_info["gpus"]]
    pair_node_ranks = f"{','.join(src_ranks)}#{','.join(dst_ranks)}"

    # Build the command for source machine (node_rank=0)
    src_cmd_parts = [
        f"cd {repo_path}",
        f"source {conda_path}",
        f"conda activate {conda_env}",
    ]

    src_dist_run_parts = []
    if nccl_socket_ifname:
        src_dist_run_parts.append(f"NCCL_SOCKET_IFNAME={nccl_socket_ifname}")
    if ib_disable:
        src_dist_run_parts.append("NCCL_IB_DISABLE=1")

    src_dist_run_parts.extend(
        [
            "python -m torch.distributed.run",
            f"--nproc_per_node={src_machine_info['num_gpus']}",
            f"--nnodes=2",
            f"--node_rank=0",
            f"--master_addr={src_machine}",
            f"--master_port=12346",
            "benchmark_gpu_bandwidth.py",
            f'--node_ranks "{pair_node_ranks}"',
            f"--parallel={parallel}",
            f"--send_msg_size={send_msg_size}",
            "--internode",
            "--warmup_iterations 5",
            "--iterations 5",
            f"--output_file {output_dir}/internode_bandwidth_results_{src_idx}_{dst_idx}.json",
        ]
    )

    src_cmd_parts.append(" ".join(src_dist_run_parts))
    src_cmd_str = " && ".join(src_cmd_parts)

    # Build command for destination machine (node_rank=1)
    dst_cmd_parts = [
        f"cd {repo_path}",
        f"source {conda_path}",
        f"conda activate {conda_env}",
    ]

    dst_dist_run_parts = []
    if nccl_socket_ifname:
        dst_dist_run_parts.append(f"NCCL_SOCKET_IFNAME={nccl_socket_ifname}")
    if ib_disable:
        dst_dist_run_parts.append("NCCL_IB_DISABLE=1")

    dst_dist_run_parts.extend(
        [
            "python -m torch.distributed.run",
            f"--nproc_per_node={dst_machine_info['num_gpus']}",
            f"--nnodes=2",
            f"--node_rank=1",
            f"--master_addr={src_machine}",
            f"--master_port=12346",
            "benchmark_gpu_bandwidth.py",
            f'--node_ranks "{pair_node_ranks}"',
            f"--parallel={parallel}",
            f"--send_msg_size={send_msg_size}",
            "--internode",
            "--warmup_iterations 5",
            "--iterations 5",
            f"--output_file {output_dir}/internode_bandwidth_results_{dst_idx}_{src_idx}.json",
        ]
    )

    dst_cmd_parts.append(" ".join(dst_dist_run_parts))
    dst_cmd_str = " && ".join(dst_cmd_parts)

    # Execute ssh commands
    print(f"Launching internode test on source machine {src_machine}...")
    src_ssh_cmd = ["ssh"]
    if ssh_key_path:
        src_ssh_cmd.extend(["-i", ssh_key_path])
    src_ssh_cmd.extend([src_machine, src_cmd_str])

    src_process = subprocess.Popen(
        src_ssh_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    print(f"Launching internode test on destination machine {dst_machine}...")
    dst_ssh_cmd = ["ssh"]
    if ssh_key_path:
        dst_ssh_cmd.extend(["-i", ssh_key_path])
    dst_ssh_cmd.extend([dst_machine, dst_cmd_str])

    dst_process = subprocess.Popen(
        dst_ssh_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    # Wait for processes to complete and get output
    print(
        f"Waiting for internode test between {src_machine} and {dst_machine} to complete..."
    )
    src_stdout, src_stderr = src_process.communicate()
    dst_stdout, dst_stderr = dst_process.communicate()

    return (
        src_stdout,
        src_stderr,
        dst_stdout,
        dst_stderr,
        src_process.returncode,
        dst_process.returncode,
    )
