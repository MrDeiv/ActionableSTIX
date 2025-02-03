from datasets import load_dataset
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
import time
from tqdm import tqdm
import json

# vector databases
from langchain_community.vectorstores import Chroma, FAISS, LanceDB

dataset = load_dataset("stanfordnlp/coqa")
MODEL = "llama3.1:8b"

# https://huggingface.co/datasets/stanfordnlp/coqa
documents = dataset['train']['story'] # 7.2k records

stores = [Chroma, FAISS, LanceDB]
results = {}
K = [1, 3, 5, 10]
batch = 2000

text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)

progress = tqdm(total=len(stores), desc="Testing stores")
for store in stores:
    progress.update(1)
    results[store.__name__] = {}

    subset = documents[:batch]
    texts = text_splitter.create_documents(subset)

    try:
        db = store.from_documents(texts, OllamaEmbeddings(model=MODEL))
        for k_star in K:
            retriever = db.as_retriever(
                search_kwargs={
                    'k': k_star
                }
            )
            start = time.time()
            retriever.get_relevant_documents(dataset['train']['story'][0])
            results[store.__name__][k_star] = time.time() - start
    except Exception as e:
        print(e)

progress.close()

with open("search_time_results.json", "w") as f:
    f.write(json.dumps(results))