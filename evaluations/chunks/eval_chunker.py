import time
import json
from transformers import AutoTokenizer
from tqdm import tqdm
from datasets import load_dataset

from langchain_text_splitters import RecursiveCharacterTextSplitter, CharacterTextSplitter, SentenceTransformersTokenTextSplitter, NLTKTextSplitter
from langchain_experimental.text_splitter import SemanticChunker as LangchainSemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings

from chonkie import TokenChunker, WordChunker, SentenceChunker, RecursiveChunker, RecursiveRules, SemanticChunker, SDPMChunker, LateChunker, Chunk

CHUNK_SIZE = 400
CHUNK_OVERLAP = 0.3
TOKENIZER_NAME = "meta-llama/Llama-3.1-8B"

if __name__ == '__main__':
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)

    ds = load_dataset("chonkie-ai/wikipedia-100k")['train']
    
    texts = [article['text'] for article in ds]
    
    #state_of_union_txt = open("state_of_the_union.txt", "r").read()
    #print(f"Loaded {len(state_of_union_txt.split(" "))} words")

    chunkers_langchain = [
        RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_SIZE*CHUNK_OVERLAP
        ),
        CharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_SIZE*CHUNK_OVERLAP
        ),
        NLTKTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_SIZE*CHUNK_OVERLAP
        ),
        """ LangchainSemanticChunker(
            embeddings=HuggingFaceEmbeddings(),
        ) """
    ]

    chunkers_chonkie = [
        TokenChunker(
            tokenizer=tokenizer,
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP
        ),
        WordChunker(
            tokenizer=tokenizer,
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP
        ),
        SentenceChunker(
            tokenizer=tokenizer,
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP
        ),
        RecursiveChunker(
            tokenizer=tokenizer,
            chunk_size=CHUNK_SIZE
        ),
        SemanticChunker(
            embedding_model="all-MiniLM-L6-v2",                
            chunk_size=CHUNK_SIZE
        ),
        SDPMChunker(
            embedding_model="all-MiniLM-L6-v2",                            
            chunk_size=CHUNK_SIZE
        ),
        LateChunker(
            embedding_model="all-MiniLM-L6-v2",
            chunk_size=CHUNK_SIZE
        )
    ]

    results = {}
    progress = tqdm(total=len(chunkers_langchain), desc="Evaluating Langchain Chunkers")
    for chunker in chunkers_langchain:
        class_name = "LC_" + chunker.__class__.__name__
        results[class_name] = {}

        # chunk the text and measure the time required
        start = time.time()
        try:
            #texts = chunker.split_text(state_of_union_txt)
            chunks = chunker.create_documents(texts)
        except Exception as e:
            print(f"\nFailed to chunk with {class_name}")
        end = time.time()
        n_chunks = len(chunks)
        results[class_name]["n_chunks"] = n_chunks
        results[class_name]["time"] = end - start

        progress.update(1)
    progress.close()

    progress = tqdm(total=len(chunkers_chonkie), desc="Evaluating Chonkie Chunkers")
    for chunker in chunkers_chonkie:
        class_name = "CH_" + chunker.__class__.__name__
        results[class_name] = {}

        # chunk the text and measure the time required
        start = time.time()
        chunks = chunker(texts)[0]
        end = time.time()
        results[class_name]["n_chunks"] = n_chunks
        results[class_name]["time"] = end - start

        progress.update(1)
    progress.close()

    with open("results2.json", "w") as f:
        json.dump(results, f)