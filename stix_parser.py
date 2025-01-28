from stix2 import parse
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import json

FILENAME = "goofy-guineapig-stix.json"

# Load and parse STIX data
json_content = open(FILENAME).read()
stix_data = parse(json_content, allow_custom=True)

stix_objects = stix_data['objects']

# Extracting all the attack patterns
attack_patterns = [obj for obj in stix_objects if obj['type'] == 'attack-pattern']

# Mitre ATT&CK Tactics
tactics = [
    "reconnaissance",
    "resource-development",
    "initial-access",
    "execution",
    "persistence",
    "privilege-escalation",
    "defense-evasion",
    "credential-access",
    "discovery",
    "lateral-movement",
    "collection",
    "command-and-control",
    "exfiltration",
    "impact"
]

# Build a dictionary
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

# Remove empty lists
attack = {k: v for k, v in attack.items() if v}

# Save attack dictionary to JSON
json.dump(attack, open('attack.json', 'w'))