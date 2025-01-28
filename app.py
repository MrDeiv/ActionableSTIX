import os
import time
import json
import torch
import logging
from transformers import AutoTokenizer, AutoModelForDocumentQuestionAnswering, pipeline

from src.DocumentStore import DocumentStore

CUDA = "cuda"
CPU = "cpu"
CONFIG_FILE = "config.json"
UPDATE_EMBEDDINGS = False

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # tic
    tic = time.time()
    logging.info(f"Time Start: {tic:.2f} s")
    
    # load config
    device = torch.device(CUDA if torch.cuda.is_available() else CPU)
    config = json.loads(open(CONFIG_FILE).read())

    # set up document store
    ds = DocumentStore(
        model_name=config["MODELS"]['SUMMARIZATION']['NAME'],
        collection_name=config["CHROMA_COLLECTION"]['NAME'],
        persist_directory=config["CHROMA_COLLECTION"]['DIR']
    )

    # ingest documents if necessary
    if UPDATE_EMBEDDINGS:
        ds.ingest(config["DOCUMENTS_DIR"])

    # query the document store
    query = "What is the name of the malware?"
    relevant_docs = ds.vector_db.similarity_search_with_relevance_scores(
        query=query,
        k=config["DOCUMENT_SEARCH"]["k"],
        score_threshold=config["DOCUMENT_SEARCH"]["THRESHOLD"]
    )

    
    tokenizer = AutoTokenizer.from_pretrained(config["MODELS"]["DOCUMENT_QA"]["TOKENIZER"])
    model = AutoModelForDocumentQuestionAnswering.from_pretrained(config["MODELS"]["DOCUMENT_QA"]["NAME"])
    doc_qa = pipeline(
        model=model,
        tokenizer=tokenizer,
        device=device,
        framework="pt")

    # query the document store
    response = doc_qa(
        question=query, 
        context=context,
        handle_impossible_answer = True)
    print(response)

    # toc
    toc = time.time()
    print(f"Time Elapsed: {toc - tic:.2f} s")

