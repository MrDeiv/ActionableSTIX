import networkx as nx
import matplotlib.pyplot as plt
import json

def draw_graph(attack: dict):
    """
    Build and draw the attack graph according to this convention:
    - Square nodes represent MITRE tactics gathered from STIX
    - Round nodes represent attack patterns related to a tactic
    """
    # Initialize the graph
    G = nx.DiGraph()

    # Build the graph
    tactics = list(attack.keys())
    round_nodes = {}
    for i, tactic in enumerate(tactics):
        G.add_node(f"square_{i}", label=tactic, shape='square')
        round_nodes[i] = []
        for j, ap in enumerate(attack[tactic]):
            G.add_node(f"round_{i}{j}", label=ap['name'], shape='round')
            G.add_edge(f"round_{i}{j}", f"square_{i}")
            round_nodes[i].append(f"round_{i}{j}")

    # Define positions for square and round nodes
    pos_square = {f"square_{i}": (i, 0) for i in range(len(tactics))}  # Horizontal line for square nodes
    pos_round = {}
    y_offset = -1
    for i, r_nodes in round_nodes.items():
        for j, r in enumerate(r_nodes):
            pos_round[r] = (i + (j - len(r_nodes) / 2) * 0.5, y_offset)

    pos = {**pos_square, **pos_round}  # Combine positions

    # Draw the graph
    plt.figure(figsize=(10, 6))

    # Draw square nodes
    square_labels = {f"square_{i}": i for i in range(len(tactics))}
    square_text_labels = {f"square_{i}": label for i, label in enumerate(tactics)}
    nx.draw_networkx_nodes(
        G, pos, nodelist=pos_square.keys(), node_shape='s', node_color='black', label=None
    )
    nx.draw_networkx_labels(
        G, pos, labels=square_labels, font_color='white', font_size=10
    )

    # Add string labels above square nodes
    for node, (x, y) in pos_square.items():
        plt.text(x, y + 0.01, square_text_labels[node], fontsize=8, ha='center', color='black')

    # Draw round nodes
    round_labels = {r: G.nodes[r]['label'] for r in G.nodes if 'round' in r}
    nx.draw_networkx_nodes(
        G, pos, nodelist=pos_round.keys(), node_shape='o', edgecolors='red', node_color='white', linewidths=1.5, label=None
    )
    nx.draw_networkx_labels(
        G, pos, labels=round_labels, font_color='black', font_size=8
    )

    # Draw edges
    nx.draw_networkx_edges(G, pos)

    plt.title("Attack graph", fontsize=14)
    plt.axis("off")
    plt.tight_layout()
    plt.show()