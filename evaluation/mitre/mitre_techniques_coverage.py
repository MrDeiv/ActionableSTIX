import json, os

if __name__ == "__main__":
    mitre_techniques = json.load(open('mitre/mitre-techniques.json', 'r'))

    techniques_coverage = {}

    for tactic in mitre_techniques:
        for technique in mitre_techniques[tactic]['techniques']:
            techniques_coverage[technique['id']] = 0

    techniques_per_tactic = {}
    for sample in os.listdir('ablation/mitre/samples'):
        stix_json = json.load(open(os.path.join('ablation/mitre/samples', sample), encoding='utf-8'))
        objects = stix_json.get('objects', [])
        
        attack_patterns = [obj for obj in objects if obj.get('type') == 'attack-pattern']
        for ap in attack_patterns:
            tech_id = ap.get('external_references', [{}])
            for ref in tech_id:
                if 'external_id' in ref and ref['external_id'].startswith('T'):
                    tech_id = ref['external_id']
                    if 'kill_chain_phases' in ap and ap['kill_chain_phases']:
                        tactic = ap['kill_chain_phases'][0]['phase_name']
                        if tactic in techniques_per_tactic:
                            techniques_per_tactic[tactic] += 1
                        else:
                            techniques_per_tactic[tactic] = 1
                    
                    if tech_id in techniques_coverage:
                        techniques_coverage[tech_id] += 1
                    else:
                        techniques_coverage[tech_id] = 1

    total = sum(techniques_coverage.values())
    covered = sum(1 for v in techniques_coverage.values() if v > 0)
    print(f"Total techniques: {total}")
    print(f"Covered techniques: {covered}/{len(techniques_coverage)}")
    print(f"Coverage: {covered/len(techniques_coverage)*100:.2f}%")

    check_num = sum(techniques_per_tactic.values())
    print(f"Check num (should match total): {check_num} - {'OK' if check_num == total else 'MISMATCH'}")
    covered_tactics = len([v for v in techniques_per_tactic.values() if v > 0])
    print(f"Total tactics with at least one technique: {covered_tactics} out of {len(mitre_techniques)}")

    json.dump(techniques_coverage, open('ablation/mitre/mitre_techniques_coverage.json', 'w'), indent=4)
    json.dump(techniques_per_tactic, open('ablation/mitre/mitre_techniques_per_tactic.json', 'w'), indent=4)