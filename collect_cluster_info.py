import argparse
import json
import subprocess
import os
import re
import time
from utils.cmd_utils import parse_internode_bandwidth_from_output, parse_intranode_bandwidth_from_output, run_internode_bandwidth_test, run_intranode_bandwidth_test  # type: ignore
from utils.collect_cluster_utils import (
    collect_cluster_info,
    parse_args,
    read_machine_file,
)


def create_machine_groups(cluster_info, ssh_key_paths):
    machine_groups = {}  # Maps machine_idx to group_id
    group_to_machines = {}  # Maps group_id to list of machine indices
    group_id = 0

    # First pass: create groups
    for machine_idx, machine_info in enumerate(cluster_info["machines"]):
        machine = machine_info["machine"]
        ssh_key = ssh_key_paths.get(machine, "")
        region = ssh_key.split("/")[-1] if ssh_key else "unknown"

        gpu_types = set(gpu["name"] for gpu in machine_info["gpus"])
        gpu_count = machine_info["num_gpus"]
        hw_identifier = f"{','.join(sorted(gpu_types))}__{gpu_count}"

        group_key = f"{region}_{hw_identifier}"

        if group_key not in group_to_machines:
            group_to_machines[group_key] = []
            group_id += 1

        group_to_machines[group_key].append(machine_idx)
        machine_groups[machine_idx] = group_key

    group_representatives = {}
    for group_key, machines in group_to_machines.items():
        group_representatives[group_key] = machines[0]

    return machine_groups, group_to_machines, group_representatives


def run_bandwidth_test(
    cluster_info,
    output_dir,
    ssh_key_paths,
    conda_paths,
    conda_envs,
    repo_paths,
    nccl_socket_ifnames,
    ib_disable=False,
    parallel=1,
    send_msg_size=10**7,
    skip_intranode=False,
    skip_internode=False,
    optimize_internode=False,
):
    os.makedirs(output_dir, exist_ok=True)
    bandwidth_results = {
        "internode_bandwidth": {},
        "intranode_bandwidth": {},
        "internode_latency": {},
        "intranode_latency": {},
    }

    # Run intranode bandwidth tests
    if not skip_intranode:
        print("\n=== Running Intranode Bandwidth Tests ===")
        processes = {}
        for machine_idx, machine_info in enumerate(cluster_info["machines"]):
            machine = machine_info["machine"]
            num_gpus = machine_info["num_gpus"]
            conda_path = conda_paths[machine]
            conda_env = conda_envs[machine]
            repo_path = repo_paths[machine]
            nccl_socket_ifname = nccl_socket_ifnames.get(machine, "")
            ssh_key_path = ssh_key_paths.get(machine)

            process = run_intranode_bandwidth_test(
                machine,
                machine_idx,
                num_gpus,
                conda_path,
                conda_env,
                repo_path,
                nccl_socket_ifname,
                ssh_key_path,
                output_dir,
                ib_disable,
                parallel,
                send_msg_size,
            )
            processes[machine_idx] = process
        
        for machine_idx, process in processes.items():   
            stdout, stderr = process.communicate()

            if stdout:
                bandwidth, latency = parse_intranode_bandwidth_from_output(stdout)
                if bandwidth > 0:
                    bandwidth_results["intranode_bandwidth"][
                        str(machine_idx)
                    ] = bandwidth
                    bandwidth_results["intranode_latency"][str(machine_idx)] = latency
                    print(
                        f"Parsed intranode bandwidth for machine {machine_idx}: {bandwidth:.2f} Gb/s, latency: {latency:.3f} ms"
                    )

    # Run internode bandwidth tests
    if len(cluster_info["machines"]) > 1 and not skip_internode:
        print("\n=== Running Internode Bandwidth Tests ===")
        internode_bandwidth = {}
        internode_latency = {}

        pairs_to_test = []

        if optimize_internode:
            print("\n=== Creating machine groups for optimized testing ===")
            machine_groups, group_to_machines, group_representatives = (
                create_machine_groups(cluster_info, ssh_key_paths)
            )

            # Print the machine grouping information
            print("\nMachine Groups:")
            for group_key, machines in group_to_machines.items():
                machine_names = [
                    cluster_info["machines"][idx]["machine"] for idx in machines
                ]
                print(f"Group '{group_key}': {machines} - {machine_names}")
                print(
                    f"  Representative: {group_representatives[group_key]} - {cluster_info['machines'][group_representatives[group_key]]['machine']}"
                )

            # Create pairs to test:
            # 1. Always test all machines within the same group
            # 2. For different groups, only test the representatives

            # First, test within each group
            for group_key, machines in group_to_machines.items():
                for i in range(len(machines)):
                    for j in range(i + 1, len(machines)):
                        pairs_to_test.append((machines[i], machines[j]))
                        print(
                            f"Adding intra-group pair: ({machines[i]}, {machines[j]})"
                        )

            # Test between group representatives
            group_keys = list(group_representatives.keys())
            for i in range(len(group_keys)):
                for j in range(i + 1, len(group_keys)):
                    rep_i = group_representatives[group_keys[i]]
                    rep_j = group_representatives[group_keys[j]]
                    pairs_to_test.append((rep_i, rep_j))
                    print(f"Adding inter-group representative pair: ({rep_i}, {rep_j})")

            print(
                f"\nOptimized testing: {len(pairs_to_test)} pairs instead of {len(cluster_info['machines']) * (len(cluster_info['machines']) - 1) // 2}"
            )
        else:
            for src_idx in range(len(cluster_info["machines"])):
                for dst_idx in range(src_idx + 1, len(cluster_info["machines"])):
                    pairs_to_test.append((src_idx, dst_idx))

        for src_idx, dst_idx in pairs_to_test:
            src_machine_info = cluster_info["machines"][src_idx]
            src_machine = src_machine_info["machine"]

            dst_machine_info = cluster_info["machines"][dst_idx]
            dst_machine = dst_machine_info["machine"]

            print(
                f"\n--- Testing INTERNODE bandwidth between {src_machine} and {dst_machine} ---"
            )

            src_conda_path = conda_paths[src_machine]
            src_conda_env = conda_envs[src_machine]
            src_repo_path = repo_paths[src_machine]
            src_nccl_socket_ifname = nccl_socket_ifnames.get(src_machine, "")
            src_ssh_key_path = ssh_key_paths.get(src_machine)

            # Run the internode bandwidth test between these two machines
            src_stdout, src_stderr, dst_stdout, dst_stderr, src_rc, dst_rc = (
                run_internode_bandwidth_test(
                    src_machine,
                    dst_machine,
                    src_idx,
                    dst_idx,
                    src_machine_info,
                    dst_machine_info,
                    src_conda_path,
                    src_conda_env,
                    src_repo_path,
                    src_nccl_socket_ifname,
                    src_ssh_key_path,
                    output_dir,
                    ib_disable,
                    parallel,
                    send_msg_size,
                )
            )

            if src_stdout:
                print(f"== Output from source machine {src_machine} ==")
                print(src_stdout)
            if src_stderr:
                print(f"== Errors from source machine {src_machine} ==")
                print(src_stderr)
            if dst_stdout:
                print(f"== Output from destination machine {dst_machine} ==")
                print(dst_stdout)
            if dst_stderr:
                print(f"== Errors from destination machine {dst_machine} ==")
                print(dst_stderr)

            print(f"Source process exited with code {src_rc}")
            print(f"Destination process exited with code {dst_rc}")

            # Parse bandwidth directly from output
            src_bandwidth, src_latency = parse_internode_bandwidth_from_output(
                src_stdout
            )
            dst_bandwidth, dst_latency = parse_internode_bandwidth_from_output(
                dst_stdout
            )

            # Initialize dictionaries if needed
            if src_idx not in internode_bandwidth:
                internode_bandwidth[src_idx] = {}
            if src_idx not in internode_latency:
                internode_latency[src_idx] = {}
            if dst_idx not in internode_bandwidth:
                internode_bandwidth[dst_idx] = {}
            if dst_idx not in internode_latency:
                internode_latency[dst_idx] = {}

            # Calculate final bandwidth and latency
            final_bandwidth = 0.0
            final_latency = 0.0

            if src_bandwidth > 0 and dst_bandwidth > 0:
                final_bandwidth = (src_bandwidth + dst_bandwidth) / 2
            elif src_bandwidth > 0:
                final_bandwidth = src_bandwidth
            elif dst_bandwidth > 0:
                final_bandwidth = dst_bandwidth

            if src_latency > 0 and dst_latency > 0:
                final_latency = (src_latency + dst_latency) / 2
            elif src_latency > 0:
                final_latency = src_latency
            elif dst_latency > 0:
                final_latency = dst_latency

            # Store results symmetrically
            internode_bandwidth[src_idx][dst_idx] = final_bandwidth
            internode_bandwidth[dst_idx][src_idx] = final_bandwidth
            internode_latency[src_idx][dst_idx] = final_latency
            internode_latency[dst_idx][src_idx] = final_latency

            print(
                f"Measured bandwidth between {src_machine} and {dst_machine}: {final_bandwidth:.2f} Gb/s"
            )
            print(
                f"Measured latency between {src_machine} and {dst_machine}: {final_latency:.3f} ms"
            )

        if optimize_internode:
            print("\n=== Propagating bandwidth results to non-tested pairs ===")
            for i in range(len(cluster_info["machines"])):
                if i not in internode_bandwidth:
                    internode_bandwidth[i] = {}
                if i not in internode_latency:
                    internode_latency[i] = {}

                for j in range(len(cluster_info["machines"])):
                    if i == j:
                        continue

                    if j in internode_bandwidth[i]:
                        continue

                    group_i = machine_groups[i]
                    group_j = machine_groups[j]

                    if group_i == group_j:
                        print(
                            f"Warning: Same-group pair ({i}, {j}) not directly tested"
                        )
                        continue

                    # For different groups, propagate from representatives
                    rep_i = group_representatives[group_i]
                    rep_j = group_representatives[group_j]

                    if rep_j in internode_bandwidth.get(rep_i, {}):
                        internode_bandwidth[i][j] = internode_bandwidth[rep_i][rep_j]
                        internode_latency[i][j] = internode_latency[rep_i][rep_j]
                        print(
                            f"Propagated results from ({rep_i}, {rep_j}) to ({i}, {j}): {internode_bandwidth[i][j]:.2f} Gb/s"
                        )
                    else:
                        print(
                            f"Warning: Missing representative results for ({rep_i}, {rep_j})"
                        )

        # Update the final bandwidth results
        bandwidth_results["internode_bandwidth"] = internode_bandwidth
        bandwidth_results["internode_latency"] = internode_latency

    return bandwidth_results


def main():
    args = parse_args()
    (
        machines,
        ssh_key_paths,
        conda_paths,
        conda_envs,
        repo_paths,
        nccl_socket_ifnames,
    ) = read_machine_file(args.machine_file)

    print(f"Collecting GPU information from {len(machines)} machines...")

    for machine in machines:
        print(f"Machine: {machine}")
        if machine in ssh_key_paths:
            print(f"  SSH key: {ssh_key_paths[machine]}")
        print(f"  Conda path: {conda_paths[machine]}")
        print(f"  Conda env: {conda_envs[machine]}")
        print(f"  Repo path: {repo_paths[machine]}")
        if machine in nccl_socket_ifnames and nccl_socket_ifnames[machine]:
            print(f"  NCCL socket interface: {nccl_socket_ifnames[machine]}")

    cluster_info = collect_cluster_info(machines, ssh_key_paths)

    print(
        f"Found {cluster_info['total_gpus']} GPUs across {len(cluster_info['machines'])} machines."
    )

    for machine_info in cluster_info["machines"]:
        machine = machine_info["machine"]
        print(f"\n{machine}: {machine_info['num_gpus']} GPUs")
        for gpu in machine_info["gpus"]:
            print(
                f"  - GPU {gpu['index']}: {gpu['name']} ({gpu['memory_total_mib']} MiB), Global Rank: {gpu['global_rank']}"
            )

    bandwidth_results = None

    # Run bandwidth tests if not skipped
    if not args.skip_bandwidth and cluster_info["total_gpus"] > 0:
        print("\nRunning bandwidth tests between GPUs...")
        bandwidth_test_start = time.perf_counter()
        bandwidth_results = run_bandwidth_test(
            cluster_info,
            args.output_dir,
            ssh_key_paths,
            conda_paths,
            conda_envs,
            repo_paths,
            nccl_socket_ifnames,
            ib_disable=args.ib_disable,
            parallel=args.parallel,
            send_msg_size=args.send_msg_size,
            skip_intranode=args.skip_intranode,
            skip_internode=args.skip_internode,
            optimize_internode=args.optimize_internode,
        )
        bandwidth_test_end = time.perf_counter()
        print(
            f"Bandwidth testing took {bandwidth_test_end - bandwidth_test_start} seconds"
        )

    if bandwidth_results:
        cluster_info["bandwidth"] = bandwidth_results

    with open(args.output, "w") as f:
        json.dump(cluster_info, f, indent=4)

    print(f"\nCluster info saved to {args.output}")


if __name__ == "__main__":
    main()
