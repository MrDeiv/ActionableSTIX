import matplotlib.pyplot as plt
import numpy as np
import json

LABEL_SIZE = 35
TITLE_SIZE = 40
TICK_SIZE = 25

"""
Execution times
"""

outs = [
    "results/case1/LOW_output.json",
    "results/case2/LOW_output.json",
    "results/case3/LOW_output.json",
    "results/case4/LOW_output.json",
    "results/case5/LOW_output.json",
]

times = [
    "results/case1/execution_times_goofy.json",
    "results/case2/execution_times_smooth.json",
    "results/case3/execution_times_sieve.json",
    "results/case4/execution_times_jaguar.json",
    "results/case5/execution_times_coldsteel.json",
]

# total time per experiment
t = []
for time in times:
    t.append(json.load(open(time)))

plt.boxplot(t)
plt.title('Evaluation Execution Times', fontsize=TITLE_SIZE)
plt.xlabel('Case #', fontsize=LABEL_SIZE)
plt.ylabel('Execution Time (s)', fontsize=LABEL_SIZE)
plt.tick_params(axis='x', labelsize=TICK_SIZE)
plt.tick_params(axis='y', labelsize=TICK_SIZE)
plt.show()

# time per step
n = []
for out in outs:
    data = json.load(open(out))
    n.append(sum([len(milestone['attack_steps']) for milestone in data]))

data = [json.load(open(time)) for time in times]

computed = []
for i,d in enumerate(data):
    computed.append([])
    for m in d:
        computed[i].append(np.array(m)/n[i])

plt.boxplot(computed)

# Add labels and title
plt.title('Evaluation Execution Times per Attack Step', fontsize=TITLE_SIZE)
plt.xlabel('Case #', fontsize=LABEL_SIZE)
plt.ylabel('Execution Time (s) per Attack Step', fontsize=LABEL_SIZE)
plt.tick_params(axis='x', labelsize=TICK_SIZE)
plt.tick_params(axis='y', labelsize=TICK_SIZE)

# Show the plot
plt.show()

"""
F1 scores
"""

scores = [
    "results/case1/execution_scores_goofy.json",
    "results/case2/execution_scores_smooth.json",
    "results/case3/execution_scores_sieve.json",
    "results/case4/execution_scores_jaguar.json",
    "results/case5/execution_scores_coldsteel.json",
]

s = []
for score in scores:
    s.append(json.load(open(score)))

f1_scores = []
for i, score in enumerate(s):
    f1_scores.append([])
    for measures in score:
        f1_scores[i].append(np.mean(measures['f1']))

plt.boxplot(f1_scores)

# Add labels and title
plt.title('Evaluation F1 Scores', fontsize=TITLE_SIZE)
plt.xlabel('Case #', fontsize=LABEL_SIZE)
plt.ylabel('F1 Score', fontsize=LABEL_SIZE)
plt.tick_params(axis='x', labelsize=TICK_SIZE)
plt.tick_params(axis='y', labelsize=TICK_SIZE)

# Show the plot
plt.show()


