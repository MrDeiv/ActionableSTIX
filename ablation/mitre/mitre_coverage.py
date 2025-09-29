import json
import os

SAMPLES = 'ablation/mitre/samples'
MITRE_TACTICS_FILE = 'mitre/mitre-tactics.json'

if __name__ == "__main__":
    tactics = json.load(open(MITRE_TACTICS_FILE, 'r'))

    tactics_coverage = {}

    for sample in os.listdir(SAMPLES):
        stix_json = json.load(open(os.path.join(SAMPLES, sample), encoding='utf-8'))
        objects = stix_json.get('objects', [])
        
        attack_patterns = [obj for obj in objects if obj.get('type') == 'attack-pattern']
        for ap in attack_patterns:
            tactic = ap['kill_chain_phases'][0]['phase_name'] if 'kill_chain_phases' in ap and ap['kill_chain_phases'] else 'N/A'
            if tactic in tactics_coverage:
                tactics_coverage[tactic] += 1
            else:
                tactics_coverage[tactic] = 1

    # add tactics with 0 coverage
    for tactic in tactics:
        if tactic not in tactics_coverage:
            tactics_coverage[tactic] = 0

    json.dump(tactics_coverage, open('ablation/mitre/mitre_coverage.json', 'w'), indent=4)
    