"""
Visualization functions for GPU graphs and partitioning results
"""

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx


def get_unique_machines(data):
    """Get a list of unique machine names and assign colors."""
    machines = set(gpu.get("machine", "Unknown") for gpu in data.get("gpu_list", []))

    # Generate colors based on number of machines
    import matplotlib.cm as cm

    colors = cm.rainbow(np.linspace(0, 1, len(machines)))

    # Create a mapping of machine to color
    return {
        machine: f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}"
        for machine, (r, g, b, _) in zip(machines, colors)
    }


def visualize_graph(G, data, title="GPU Graph", output_file=None):
    """Visualize the graph with edge weights."""
    if G.number_of_nodes() == 0:
        print("Error: Graph has no nodes to visualize.")
        return

    plt.figure(figsize=(12, 10))

    # Position nodes using spring layout
    pos = nx.spring_layout(G, seed=42)

    # Get edge weights for width and color
    weights = [G[u][v].get("weight", 0.5) for u, v in G.edges()]
    max_weight = max(weights) if weights else 1.0
    normalized_weights = [w / max_weight for w in weights]

    # Draw nodes - color by machine
    machine_colors = get_unique_machines(data)

    # Safely get node colors, defaulting to gray if machine not found
    node_colors = []
    for node in G.nodes():
        machine = G.nodes[node].get("machine", "Unknown")
        node_colors.append(machine_colors.get(machine, "#cccccc"))

    # Draw the graph elements
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=700)
    nx.draw_networkx_edges(
        G, pos, width=[max(0.5, w * 5) for w in normalized_weights], edge_color="black"
    )

    # Add labels
    labels = {}
    for node in G.nodes():
        name = G.nodes[node].get("name", "Unknown").split()[-1]
        memory = G.nodes[node].get("memory_total_mib", 0) / 1024
        labels[node] = f"{name}\n{memory:.1f}GB"

    nx.draw_networkx_labels(G, pos, labels, font_size=8)

    # Add edge weight labels
    edge_labels = {(u, v): f"{G[u][v].get('weight', 0):.2f}" for u, v in G.edges()}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=7)

    # Add a legend for machine colors
    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor=color, label=machine)
        for machine, color in machine_colors.items()
    ]
    plt.legend(handles=legend_elements, loc="upper left", title="Machines")

    plt.title(title)
    plt.axis("off")
    plt.tight_layout()

    # Save the figure if output file is specified
    if output_file:
        plt.savefig(output_file, dpi=300)
    else:
        # Default filename based on title
        plt.savefig(f"{title.replace(' ', '_')}.png", dpi=300)

    plt.show()


def visualize_partitioned_graph(
    G, partitions, data, title="Partitioned GPU Graph", output_file=None
):
    """Visualize the graph with partitions highlighted."""
    if G.number_of_nodes() == 0:
        print("Error: Graph has no nodes to visualize.")
        return

    plt.figure(figsize=(14, 12))

    # Position nodes using spring layout
    pos = nx.spring_layout(G, seed=42)

    # Get number of partitions for color mapping
    num_partitions = len(set(partitions.values()))
    import matplotlib.cm as cm

    partition_colors = cm.tab20(np.linspace(0, 1, num_partitions))

    # Get edge weights for width
    weights = [G[u][v].get("weight", 0.5) for u, v in G.edges()]
    max_weight = max(weights) if weights else 1.0
    normalized_weights = [w / max_weight for w in weights]

    # Draw edges - highlight edges between partitions
    cut_edges = [(u, v) for u, v in G.edges() if partitions[u] != partitions[v]]
    within_edges = [(u, v) for u, v in G.edges() if partitions[u] == partitions[v]]

    # Draw within-partition edges
    for u, v in within_edges:
        idx = list(G.edges()).index((u, v))
        nx.draw_networkx_edges(
            G,
            pos,
            edgelist=[(u, v)],
            width=normalized_weights[idx] * 5,
            edge_color="gray",
            alpha=0.7,
        )

    # Draw between-partition edges
    for u, v in cut_edges:
        idx = list(G.edges()).index((u, v))
        nx.draw_networkx_edges(
            G,
            pos,
            edgelist=[(u, v)],
            width=normalized_weights[idx] * 5,
            edge_color="red",
            style="dashed",
        )

    # Draw nodes - color by partition
    for part in range(num_partitions):
        node_list = [node for node, p in partitions.items() if p == part]
        if not node_list:
            continue

        color = partition_colors[part]
        nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=node_list,
            node_color=[
                f"#{int(color[0]*255):02x}{int(color[1]*255):02x}{int(color[2]*255):02x}"
            ],
            node_size=700,
            label=f"Partition {part}",
        )

    # Add labels
    labels = {}
    for node in G.nodes():
        name = G.nodes[node].get("name", "Unknown").split()[-1]  # Just the model name
        machine = G.nodes[node].get("machine", "Unknown")
        memory = G.nodes[node].get("memory_total_mib", 0) / 1024
        labels[node] = (
            f"{name}\n{memory:.1f}GB\n{machine[-3:]}"  # Last 3 chars of machine name
        )

    nx.draw_networkx_labels(G, pos, labels, font_size=8)

    # Add a legend for partitions
    plt.legend(title="Partitions", loc="upper right")

    plt.title(title)
    plt.axis("off")
    plt.tight_layout()

    # Save the figure if output file is specified
    if output_file:
        plt.savefig(output_file, dpi=300)
    else:
        # Default filename based on title
        plt.savefig(f"{title.replace(' ', '_')}.png", dpi=300)

    plt.show()


def visualize_partitions_with_metrics(
    G, partitions, metrics, data, title="GPU Partitions with Metrics", output_file=None
):
    """Visualize partitions with key metrics displayed."""
    # Create a figure with two subplots: graph and metrics
    fig = plt.figure(figsize=(16, 10))

    # Graph subplot
    graph_ax = plt.subplot2grid((1, 5), (0, 0), colspan=4)

    # Position nodes using spring layout
    pos = nx.spring_layout(G, seed=42)

    # Get number of partitions for color mapping
    num_partitions = len(set(partitions.values()))
    import matplotlib.cm as cm

    partition_colors = cm.tab20(np.linspace(0, 1, num_partitions))

    # Draw nodes colored by partition
    for part in range(num_partitions):
        node_list = [node for node, p in partitions.items() if p == part]
        if not node_list:
            continue

        color = partition_colors[part]
        nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=node_list,
            ax=graph_ax,
            node_color=[
                f"#{int(color[0]*255):02x}{int(color[1]*255):02x}{int(color[2]*255):02x}"
            ],
            node_size=700,
            label=f"Partition {part}",
        )

    # Draw edges - highlight edges between partitions
    cut_edges = [(u, v) for u, v in G.edges() if partitions[u] != partitions[v]]
    within_edges = [(u, v) for u, v in G.edges() if partitions[u] == partitions[v]]

    # Draw within-partition edges
    nx.draw_networkx_edges(
        G,
        pos,
        edgelist=within_edges,
        ax=graph_ax,
        width=1.0,
        edge_color="gray",
        alpha=0.7,
    )

    # Draw between-partition edges
    nx.draw_networkx_edges(
        G,
        pos,
        edgelist=cut_edges,
        ax=graph_ax,
        width=2.0,
        edge_color="red",
        style="dashed",
    )

    # Add labels
    labels = {}
    for node in G.nodes():
        name = G.nodes[node].get("name", "Unknown").split()[-1]
        part = partitions[node]
        labels[node] = f"{name}\nP{part}"

    nx.draw_networkx_labels(G, pos, labels, ax=graph_ax, font_size=8)

    # Add a legend
    graph_ax.legend(title="Partitions", loc="upper right")

    graph_ax.set_title(title)
    graph_ax.axis("off")

    # Metrics subplot
    metrics_ax = plt.subplot2grid((1, 5), (0, 4), colspan=1)
    metrics_ax.axis("off")

    # Display key metrics
    metrics_text = f"""
    Partitioning Metrics:
    
    Number of partitions: {num_partitions}
    
    Cut value: {metrics.get('cut_weight', 'N/A'):.2f}
    
    Internal weight: {metrics.get('internal_weight', 'N/A'):.2f}
    
    Cut ratio: {metrics.get('ratio_cut_to_total', 'N/A'):.2f}
    
    Balance: {metrics.get('balance', 'N/A'):.2f}
    
    Partition sizes:
    """

    for part, size in metrics.get("partition_sizes", {}).items():
        metrics_text += f"   P{part}: {size} nodes\n"

    metrics_ax.text(0, 0.5, metrics_text, fontsize=10, va="center")

    plt.tight_layout()

    # Save the figure if output file is specified
    if output_file:
        plt.savefig(output_file, dpi=300)
    else:
        # Default filename based on title
        plt.savefig(f"{title.replace(' ', '_')}.png", dpi=300)

    plt.show()
