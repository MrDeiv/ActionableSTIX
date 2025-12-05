import os, json
import uuid

DT_FOLDER = 'ablation/mitre/samples'

if __name__ == "__main__":

    dt = []
    for stix in os.listdir(DT_FOLDER):
        item = {}
        stix_json = json.load(open(os.path.join(DT_FOLDER, stix), encoding='utf-8'))
        objects = stix_json.get('objects', [])
        
        item['id'] = str(uuid.uuid4())

        software = [obj for obj in objects if obj.get('type') == 'software']
        for sw in software:
            name = sw.get('name', 'N/A').lower()
            types = sw['extensions']['software-types-ext']['software_types'] if 'extensions' in sw and 'software-types-ext' in sw['extensions'] else []
            
            if 'operation-system' in types:
                item['os'] = name
            else:
                item['os'] = 'N/A'
        
        identity = [obj for obj in objects if obj.get('type') == 'identity']
        if identity:
            item['sha256'] = identity[0]['external_references'][0]['hashes']['SHA-256'] if 'external_references' in identity[0] and 'hashes' in identity[0]['external_references'][0] else 'N/A'
            item['external_id'] = identity[0]['external_references'][0]['external_id'] if 'external_references' in identity[0] else 'N/A'
            item['external_src'] = identity[0].get('name', 'N/A')
            item['labels'] = identity[0].get('labels', [])

        analysis = [obj for obj in objects if obj.get('type') == 'malware-analysis']
        # select the most severe analysis
        results_priorities = {'malicious': 3, 'suspicious': 2, 'benign': 1, 'unknown': 0}
        if analysis:
            analysis = sorted(analysis, key=lambda x: results_priorities.get(x.get('result', 'unknown'), 0), reverse=True)
            
            # take the most severe label
            item['verdict'] = analysis[0].get('result', 'unknown')

            # among the ones with the same label, take the most recent
            same_verdict = [a for a in analysis if a.get('result', 'unknown') == item['verdict']]
            same_verdict = sorted(same_verdict, key=lambda x: x.get('analysis_started', 'N/A'), reverse=True)
            item['analysis_date'] = same_verdict[0].get('analysis_started', 'N/A')
        else:
            item['verdict'] = 'unknown'
            item['analysis_date'] = 'N/A'

        dt.append(item)
    
    json.dump(dt, open('ablation/mitre/stix_dataset.json', 'w'), indent=4)
