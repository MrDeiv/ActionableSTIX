import json
import networkx as nx

def compute_graph_distance(G, src_id, tgt_id):
    try:
        length = nx.shortest_path_length(G, source=src_id, target=tgt_id)
        path = nx.shortest_path(G, source=src_id, target=tgt_id)
        return length, path
    except nx.NetworkXNoPath:
        return float('inf'), []

if __name__ == '__main__':
    mitre_graph = json.load(open('mitre/mitre_graph.json', 'r'))
    G = nx.node_link_graph(mitre_graph)

    src_id = 'T1595.001'
    tgt_id = 'T1595.002'

    distance, path = compute_graph_distance(G, src_id, tgt_id)
    print(f"Distance between {src_id} and {tgt_id}: {distance}")