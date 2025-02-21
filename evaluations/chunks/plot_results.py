import json
from matplotlib import pyplot as plt

results = json.load(open("results.json"))

models = list(results.keys())
time = [results[model]['time'] for model in models]

fig, axs = plt.subplots(1, 1)

# time
axs.bar(models, time)
axs.set_title("Time to Chunk")

# set label inclinations
axs.set_xticklabels(models, rotation=45, ha='right')
axs.set_ylabel("Time (s)")

plt.tight_layout()
plt.show()