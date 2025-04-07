import json
import os

if __name__ == "__main__":
    ref = "out/LOW_output.json"
    gt = "ground_truths/coldsteel.json"

    if os.path.exists(gt):
        print(f"File {gt} already exists. Skipping.")
        exit(0)

    ref_data = json.load(open(ref))
    gt_data = []

    for milestone in ref_data:
        tmp = {}
        tmp['attack_steps'] = []
        for attack_step in milestone['attack_steps']:
            tmp['attack_steps'].append({
                'name': attack_step['name'],
                'truth': ''
            })

        gt_data.append(tmp)

    with open(gt, 'w') as f:
        json.dump(gt_data, f, indent=4)
    
