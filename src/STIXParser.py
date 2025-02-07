import json
from stix2 import parse

class STIXParser:
    def __init__(self):
        self.objects = None
    
    def parse(self, filename: str):
        content = open(filename).read()
        data = parse(content, allow_custom=True)
        self.objects = data['objects']

    def extract_attack_patterns(self):
        attack_patterns = [obj for obj in self.objects if obj['type'] == 'attack-pattern']
        return attack_patterns
    
    def extract_malware(self):
        malware = [obj for obj in self.objects if obj['type'] == 'malware']
        return malware
    
    def extract_indicators(self):
        indicators = [obj for obj in self.objects if obj['type'] == 'indicator']
        return indicators
    
    def _stringify_malware(self, malware):
        text = f"""
        Malware Name: {malware['name']} | Malware Description: {malware['description']}
        """

        keys = ['malware_types', 'capabilities', 'operating_system_refs', 'architecture_execution_envs', 'implementation_languages', 'x_operating_systems']
        text += " | ".join([f"Malware {key}: {" ".join(malware[key])}" for key in keys if key in malware])

        return text 
    
    def _stringify_indicator(self, indicator):
        text = f"""
        Indicator Name: {indicator['name']} | Indicator Description: {indicator['description']}
        """

        keys = ['pattern', 'kill_chain_phases']
        text += " | ".join([f"Indicator {key}: {indicator[key]}\n" for key in keys if key in indicator and 'rule' not in indicator[key]])

        return text
    
    def stringify_object(self, obj):
        """
        Stringify object
        """
        f = getattr(self, f"_stringify_{obj['type']}", None)
        return f(obj)