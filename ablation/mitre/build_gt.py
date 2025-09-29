import os, json
import networkx as nx

SAMPLES = 'ablation/mitre/samples'
MITRE_GRAPH_FILE = 'mitre/mitre_graph.json'

if __name__ == "__main__":
    
    G = nx.node_link_graph(json.load(open(MITRE_GRAPH_FILE, 'r')))
    gt = []
    for sample in os.listdir(SAMPLES):
        stix = json.load(open(os.path.join(SAMPLES, sample), encoding='utf-8'))
        objects = stix.get('objects', [])
        for obj in objects:
            if obj.get('type') == 'attack-pattern':
                external_refs = obj.get('external_references', [])
                tactic = obj.get('kill_chain_phases', ['N/A'])[0]['phase_name'] if obj.get('kill_chain_phases') else 'N/A'
                for ref in external_refs:
                    if 'source_name' in ref and ref['source_name'] == 'mitre-attack':
                        technique_id = ref.get('external_id', 'N/A')

                        if technique_id == 'T1547.011':
                            technique_id = 'T1647'
                        if technique_id == 'T1053.001':
                            technique_id = 'T1053.002'

                        if technique_id in G:
                            name = G.nodes[technique_id].get('name', 'N/A')
                            description = G.nodes[technique_id].get('description', 'N/A')
                            gt.append({
                                'mitre_technique': technique_id,
                                'mitre_tactic': tactic,
                                'name': name,
                                'description': description
                            })
                        else:
                            print("No technique found in the graph for ID:", technique_id)
    print("Number of ground truth cases:", len(gt))
    json.dump(gt, open('ablation/mitre/gt.json', 'w'), indent=4)