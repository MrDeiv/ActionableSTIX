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