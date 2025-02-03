# plot
import matplotlib.pyplot as plt
import json
from langchain_community.vectorstores import Chroma, FAISS, LanceDB

r = json.loads(open("load_time_results.json").read())
plt.figure(figsize=(10, 5))
plt.xlabel("Documents Ingested")
plt.ylabel("Time (minutes)")
plt.title("Load Time per Vector DB")
plt.grid()

stores = [Chroma, FAISS, LanceDB]
batches = [1, 100, 500, 1000, 2000, 5000, 7000]

for store in stores:
    times = [r[store.__name__][str(batch)] for batch in batches]

    # to minutes
    times = [time / 60 for time in times]
    plt.plot(batches, times, label=store.__name__)

plt.legend()
plt.savefig("load_time_results.png")