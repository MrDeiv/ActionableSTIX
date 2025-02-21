from langchain_text_splitters import NLTKTextSplitter
import time
import json

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_pinecone import PineconeVectorStore
import os
from tqdm import tqdm
from datasets import load_dataset
import numpy as np

CHUNK_SIZE = 400
CHUNK_OVERLAP = 0.3

if __name__ == '__main__':

    ks = [1, 3, 4, 5, 10]

    stores = [
        FAISS,
        Chroma,
        #PineconeVectorStore
    ]
    ds = load_dataset("stepkurniawan/sustainability-methods-wiki", "50_QA")
    documents = ds['train']['contexts']

    splitter = NLTKTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_SIZE*CHUNK_OVERLAP
    )

    chunks = splitter.create_documents(documents)
    embeddings = HuggingFaceEmbeddings()
    questions = ds['train']['question']

    results = {}
    progress = tqdm(total=len(stores), desc="Evaluating Stores")
    for store in stores:
        results[store.__name__] = {}
        db = store.from_documents(chunks, embeddings)

        for K in ks:
            results[store.__name__][str(K)] = {}
            times = []
            for question in questions:
                start = time.time()
                db.similarity_search(question, k=K)
                end = time.time()
                times.append(end - start)
        
            mean_search_time = np.mean(times)
            results[store.__name__][str(K)]["search_time"] = mean_search_time
        progress.update(1)
    progress.close()

    with open("db_eval_k.json", "w") as f:
        json.dump(results, f)

