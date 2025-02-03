# plot
import matplotlib.pyplot as plt
import json
from langchain_community.vectorstores import Chroma, FAISS, LanceDB

r = json.loads(open("search_time_results.json").read())
plt.figure(figsize=(10, 5))
plt.xlabel("K")
plt.ylabel("Time (seconds)")
plt.title("Search Time per Vector DB @ K")
plt.grid()

stores = [Chroma, FAISS, LanceDB]
K = [1, 3, 5, 10]

try:
    for store in stores:
        times = [r[store.__name__][str(k)] for k in K]

        # to minutes
        #times = [time / 60 for time in times]
        plt.plot(K, times, label=store.__name__)
except:
    pass

plt.legend()
plt.savefig("search_time_results.png")