import json, os
import networkx as nx

if __name__ == '__main__':

    G = nx.Graph()

    mitre = json.load(open('mitre/mitre-techniques.json', 'r'))

    root = G.add_node("root_node", type="root", name="MITRE ATT&CK")

    for tactic in mitre:
        node = G.add_node(mitre[tactic]['id'], type="tactic", name=mitre[tactic]['name'], description=mitre[tactic]['description'])
        G.add_edge("root_node", mitre[tactic]['id'])

        for technique in mitre[tactic]['techniques']:
            if '.' not in technique['id']:
                tech_node = G.add_node(technique['id'], type="technique", name=technique['name'], description=technique['description'])
                G.add_edge(mitre[tactic]['id'], technique['id'])
            else:
                subtech_node = G.add_node(technique['id'], type="sub-technique", name=technique['name'], description=technique['description'])
                parent_id = technique['id'].split('.')[0]
                G.add_edge(parent_id, technique['id'])

    json.dump(nx.node_link_data(G), open('mitre/mitre_graph.json', 'w'), indent=4)