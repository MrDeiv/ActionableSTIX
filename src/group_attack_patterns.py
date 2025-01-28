def group_attack_patterns(tactics, attack_patterns):
    """
    Group attack patterns by MITRE tactic
    """
    attack = {}

    for tactic in tactics:
        attack[tactic] = []

        for ap in attack_patterns:
            if ap['kill_chain_phases'][0]['phase_name'] == tactic:
                attack[tactic].append({
                    "name": ap['name'],
                    "description": ap['description'],
                    #"external_references": ap['external_references'] if 'external_references' in ap else None,
                    "x_mitre_detection": ap['x_mitre_detection'] if 'x_mitre_detection' in ap else None,
                    "x_mitre_platforms": ap['x_mitre_platforms'] if 'x_mitre_platforms' in ap else None,
                    "x_mitre_permissions_required": ap['x_mitre_permissions_required'] if 'x_mitre_permissions_required' in ap else None,
                    "x_mitre_data_sources": ap['x_mitre_data_sources'] if 'x_mitre_data_sources' in ap else None,
                    "x_mitre_defense_bypassed": ap['x_mitre_defense_bypassed'] if 'x_mitre_defense_bypassed' in ap else None,
                })

    attack = {k: v for k, v in attack.items() if v}
    return attack