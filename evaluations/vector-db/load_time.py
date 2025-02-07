from datasets import load_dataset
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
import time
from tqdm import tqdm
import json

# vector databases
from langchain_community.vectorstores import Chroma, FAISS, InMemoryVectorStore

dataset = load_dataset("stanfordnlp/coqa")
MODEL = "llama3.1:8b"

# https://huggingface.co/datasets/stanfordnlp/coqa
documents = dataset['train']['story'] # 7.2k records

stores = [Chroma, FAISS, InMemoryVectorStore]
results = {}
batches = [1, 100, 500, 1000, 2000, 5000, 7000]


text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)

progress = tqdm(total=len(stores), desc="Testing stores")
for store in stores:
    progress.update(1)
    results[store.__name__] = {}

    batch_progress = tqdm(total=len(batches), desc="Testing batches")
    for batch in batches:
        batch_progress.update(1)
        subset = documents[:batch]
        texts = text_splitter.create_documents(subset)

        start = time.time()
        try:
            db = store.from_documents(texts, OllamaEmbeddings(model=MODEL))
        except Exception as e:
            print(e)
        results[store.__name__][batch] = time.time() - start
    batch_progress.close()

progress.close()

with open("load_time_results.json", "w") as f:
    f.write(json.dumps(results))