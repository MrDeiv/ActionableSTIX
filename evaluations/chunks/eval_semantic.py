import time
import json
from transformers import AutoTokenizer
from tqdm import tqdm

from langchain_text_splitters import RecursiveCharacterTextSplitter, CharacterTextSplitter, SentenceTransformersTokenTextSplitter, NLTKTextSplitter
from langchain_experimental.text_splitter import SemanticChunker as LangchainSemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings

from chonkie import TokenChunker, WordChunker, SentenceChunker, RecursiveChunker, RecursiveRules, SemanticChunker, SDPMChunker, LateChunker, Chunk

TOKENIZER_NAME = "meta-llama/Llama-3.1-8B"
if __name__ == '__main__':
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)

    state_of_union_txt = open("state_of_the_union.txt", "r").read()

    lc_chunkers = [
        NLTKTextSplitter
    ]

    ch_chunkers = [
        SemanticChunker,
        SDPMChunker,
        LateChunker
    ]

    chunk_batches = [100, 200, 400, 500, 1000]
    chunks_overlaps = [0.1, 0.2, 0.3, 0.4, 0.5]

    results = {}
    progress = tqdm(total=len(lc_chunkers), desc="Evaluating Langchain")
    for chunker in lc_chunkers:
        results[chunker.__name__] = {}
        for batch in chunk_batches:
            for overlap in chunks_overlaps:
                try:
                    results[chunker.__name__][f"{batch}-{overlap}"] = {}
                    chunker_instance = chunker(
                        chunk_size=batch,
                        chunk_overlap=batch*overlap
                    )
                    start = time.time()
                    chunks = chunker_instance.create_documents([state_of_union_txt])
                    end = time.time()
                    results[chunker.__name__][f"{batch}-{overlap}"]["time"] = end-start
                    results[chunker.__name__][f"{batch}-{overlap}"]["n_chunks"] = len(chunks)
                except Exception as e:
                    print(f"error: {e}")
                    results[chunker.__name__][f"{batch}-{overlap}"] = None
        progress.update(1)
    progress.close()

    progress = tqdm(total=len(ch_chunkers), desc="Evaluating Chonkie")
    for chunker in ch_chunkers:
        results[chunker.__name__] = {}
        for batch in chunk_batches:
            for overlap in chunks_overlaps:
                try:
                    results[chunker.__name__][f"{batch}-{overlap}"] = {}
                    chunker_instance = chunker(
                        embedding_model="all-MiniLM-L6-v2",
                        chunk_size=batch
                    )
                    start = time.time()
                    chunks = chunker_instance([state_of_union_txt])
                    end = time.time()
                    results[chunker.__name__][f"{batch}-{overlap}"]["time"] = end-start
                    results[chunker.__name__][f"{batch}-{overlap}"]["n_chunks"] = len(chunks)
                except Exception as e:
                    print(f"error: {e}")
                    results[chunker.__name__][f"{batch}-{overlap}"] = None
        progress.update(1)
    progress.close()

    with open("eval_semantic.json", "w") as f:
        json.dump(results, f)