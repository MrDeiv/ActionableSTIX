import json
from langchain_text_splitters import RecursiveJsonSplitter,CharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_ollama import OllamaEmbeddings
from langchain_community.document_loaders import TextLoader, PyMuPDFLoader, UnstructuredCSVLoader
import pandas as pd
import yaml

# file = json.load(open("documents/vt_graph.json"))
#file = open("documents/ioc_anyrun.txt").read()

# file = pd.read_csv("documents/NCSC-MAR-Goofy-Guineapig-indicators.csv")
# join all columns in a row into a single string (CSV)
# rows = file.apply(lambda x: " ".join(x), axis=1).to_list()

""" data = yaml.safe_load(file)

# convert to json
j = json.loads(json.dumps(data, default=str))
splitter = RecursiveJsonSplitter()
chunks = splitter.split_json(j)
print(len(chunks)) """

""" file = TextLoader("documents/ioc_anyrun.txt").load()
text_splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=200, chunk_overlap=20)
chunks = text_splitter.split_documents(file)
print(chunks) """

""" file = PyMuPDFLoader("documents/NCSC-MAR-Goofy-Guineapig.pdf").load()
text_splitter = SemanticChunker(OllamaEmbeddings(model="llama3.1:8b"))
chunks = text_splitter.split_documents(file)
print(chunks) """