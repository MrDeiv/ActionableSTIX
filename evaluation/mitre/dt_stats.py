import json
import os

if __name__ == "__main__":
    dt = json.load(open('ablation/mitre/stix_dataset.json', 'r'))
    print(f"Number of samples in dataset: {len(dt)}")
    
    stats = {
        "samples_total": len(dt),
        'attack_patterns_total': 0,
    }

    # count malicious, benign, unknown
    verdict_counts = {"verdict_malicious": 0, "verdict_benign": 0, "verdict_suspicious": 0, "verdict_unknown": 0}
    os_counts = {'os_windows': 0, 'os_linux': 0, 'os_unknown': 0}
    months = {'upl_june': 0, 'upl_july': 0, 'upl_august': 0, 'upl_september': 0, 'upl_unknown': 0}
    labels = {}
    for item in dt:
        verdict = item.get('verdict', 'unknown')
        label = "verdict_"+verdict
        if label in verdict_counts:
            verdict_counts[label] += 1
        else:
            verdict_counts['verdict_unknown'] += 1
        
        os_type = item.get('os', 'N/A')
        os_type = os_type.split()[0] if os_type != 'N/A' else 'N/A'
        if os_type == 'windows':
            os_counts['os_windows'] += 1
        elif os_type != 'N/A':
            os_counts['os_linux'] += 1
        else:
            os_counts['os_unknown'] += 1

        created = item.get('analysis_date', 'N/A')
        if created != 'N/A':
            month = created[5:7]
            if month == '06':
                months['upl_june'] += 1
            elif month == '07':
                months['upl_july'] += 1
            elif month == '08':
                months['upl_august'] += 1
            elif month == '09':
                months['upl_september'] += 1
            else:
                months['upl_unknown'] += 1

        # count the occurrence of the labels
        for label in item.get('labels', []):
            labels[label] = labels.get(label, 0) + 1
            
    samples = 'ablation/mitre/samples'
    for stix in os.listdir(samples):
        file = json.load(open(os.path.join(samples, stix), encoding='utf-8'))
        objects = file.get('objects', [])
        for obj in objects:
            if obj.get('type') == 'attack-pattern':
                stats['attack_patterns_total'] += 1

    stats.update({'labels': labels})
    stats.update(verdict_counts)
    stats.update(os_counts)
    stats.update(months)

    json.dump(stats, open('ablation/mitre/dt_stats.json', 'w'), indent=4)
