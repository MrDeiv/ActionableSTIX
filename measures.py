import subprocess
import time
import json
from evaluate import load

SCRIPT_TO_RUN = "app.py"
NUM_RUNS = 10
GROUND_TRUTH = "ground_truths/smooth.json"
GENERATED_OUTPUT = "out/LOW_output.json"
MEASURES_TIME = "out/execution_times_smooth.json"
MEASURES_SCORE = "out/execution_scores_smooth.json"

execution_times = []
execution_scores = []

bertscore = load("bertscore")

for i in range(NUM_RUNS):
    print(f"Running {SCRIPT_TO_RUN} - Iteration {i+1}/{NUM_RUNS}")
    start_time = time.time()
    
    # Run the target script
    result = subprocess.run(["python", SCRIPT_TO_RUN], capture_output=True, text=True)
    
    end_time = time.time()
    duration = end_time - start_time
    execution_times.append(duration)

    ref_data = json.load(open(GENERATED_OUTPUT))
    gt_data = json.load(open(GROUND_TRUTH))

    scores = []
    references = []
    predictions = []
    for i, milestone in enumerate(ref_data):
        for j, attack_step in enumerate(milestone['attack_steps']):
            references.append(gt_data[i]['attack_steps'][j]['truth'])
            predictions.append(attack_step['description'])

    scores = bertscore.compute(predictions=predictions, references=references, lang="en")
    execution_scores.append(scores)
    
    print(f"Execution {i+1} took {duration:.4f} seconds. Scores: {scores}")

with open(MEASURES_TIME, "w") as f:
    json.dump(execution_times, f)

with open(MEASURES_SCORE, "w") as f:
    json.dump(execution_scores, f)

print("\nSummary:")
print(f"Total execution time: {sum(execution_times):.4f} seconds")
print(f"Average execution time: {sum(execution_times)/NUM_RUNS:.4f} seconds")
