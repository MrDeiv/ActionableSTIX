import json
from langchain_text_splitters import RecursiveJsonSplitter,CharacterTextSplitter
import pandas as pd
import yaml

# file = json.load(open("documents/vt_graph.json"))
file = open("documents/win_system_malware_goofy_guineapig_service_persistence.yml").read()

# file = pd.read_csv("documents/NCSC-MAR-Goofy-Guineapig-indicators.csv")
# join all columns in a row into a single string (CSV)
# rows = file.apply(lambda x: " ".join(x), axis=1).to_list()

data = yaml.safe_load(file)
j = json.dumps(dict(data), default=str)
print(j)
splitter = RecursiveJsonSplitter()
chunks = splitter.split_json(j)
print(chunks)