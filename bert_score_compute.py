from evaluate import load
import json
import numpy as np

if __name__ == '__main__':

    bertscore = load("bertscore")

    ref = "out/output.json"
    gt = "ground_truths/goofy.json"

    ref_data = json.load(open(ref))
    gt_data = json.load(open(gt))

    scores = []
    references = []
    predictions = []
    for i, milestone in enumerate(ref_data):
        for j, attack_step in enumerate(milestone['attack_steps']):
            references.append(gt_data[i]['attack_steps'][j]['truth'])
            predictions.append(attack_step['description'])

    scores = bertscore.compute(predictions=predictions, references=references, lang="en")
    print(scores)

    with open('out/bert_scores.json', 'w') as f:
        json.dump(scores, f, indent=4)

    print("Mean F1: ", np.mean(scores['f1']))
    print("Mean Precision: ", np.mean(scores['precision']))
    print("Mean Recall: ", np.mean(scores['recall']))
