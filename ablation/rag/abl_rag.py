from ragas import SingleTurnSample
from ragas.metrics import NonLLMContextPrecisionWithReference, NonLLMContextRecall
from datasets import load_dataset, Dataset
import numpy as np
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from DocumentStore import DocumentStore
from chunking_evaluation.chunking import ClusterSemanticChunker
from chromadb.utils import embedding_functions
from langchain_core.documents import Document
from tqdm import tqdm
import json

BM25_KS = [1, 3, 5, 10, 15, 20, 30, 40, 50, 100]
KS = BM25_KS

if __name__ == "__main__":
    # Load SQuAD dataset
    ds = load_dataset("rajpurkar/squad", split="train")
    print(f"Number of samples in training set: {len(ds)}")

    # Define metric
    metrics = [
        NonLLMContextPrecisionWithReference(),
        NonLLMContextRecall()
    ]

    default_ef = embedding_functions.DefaultEmbeddingFunction()
    chunker = ClusterSemanticChunker(default_ef, max_chunk_size=200)

    chunks = set()
    references = []
    
    titles = []
    dataset = []
    i = 0
    for item in ds:
        if item['title'] not in titles:
            titles.append(item['title'])
            i += 1
            dataset.append(item)
    ds = Dataset.from_list(dataset)
    print(f"Number of unique titles in training set: {len(titles)}")

    ds.to_json("squad_subset.json")

    progress_bar = tqdm(total=len(ds), desc="Processing samples")
    for item in ds:
        context = item['context']
        question = item['question']
        answer = item['answers']['text'][0] if item['answers']['text'] else ""
        assert answer, "No answer found in the dataset sample."
        chunked_context = chunker.split_text(context)
        for chunk in chunked_context:
            chunks.add(chunk)
        references.append(answer)
        progress_bar.update(1)
    progress_bar.close()
    print(f"Total chunks created: {len(chunks)}")

    chunks = list(chunks)

    with open("contexts.json", "w") as f:
        json.dump({
            "N": len(chunks),
            "chunks": chunks
        }, f, indent=4)

    # Experiment loop
    np.random.shuffle(chunks)

    # BM25
    exp_bm25 = []
    for bm25_k in BM25_KS:
        print(f"Evaluating BM25 with k={bm25_k}")
        bm25_retriever = BM25Retriever.from_texts(chunks)
        bm25_retriever.k = bm25_k

        expp = {
            "BM25_k": bm25_k,
        }

        for metric in metrics:
            expp[metric.name] = 0
            results = []
            progress_bar = tqdm(total=len(ds), desc=f"Evaluating {metric.name}")
            for item in ds:
                retrieved_docs = bm25_retriever.invoke(item['question'])
                retrieved_texts = [doc.page_content for doc in retrieved_docs]
                        
                reference_context = chunker.split_text(item['context'])
                result = metric.single_turn_score(
                    SingleTurnSample(
                        retrieved_contexts=retrieved_texts,
                        reference_contexts=reference_context,
                    )
                )
                results.append(result)
                progress_bar.update(1)
            progress_bar.close()
            expp[metric.name] = np.mean(results)
        exp_bm25.append(expp)

    with open("abl_rag_bm25.json", "w") as f:
        json.dump(exp_bm25, f, indent=4)

    # Vector store
    exp_vs = []
    for k in KS:
        print(f"Evaluating Vector Store with k={k}")
        docstore = DocumentStore(k=k)
        documents = [Document(page_content=chunk) for chunk in chunks]
        docstore.ingest(documents)
        expp = {
            "k": k,
        }
        for metric in metrics:
            expp[metric.name] = 0
            results = []
            progress_bar = tqdm(total=len(ds), desc=f"Evaluating {metric.name}")
            for item in ds:
                retrieved_docs = docstore.retriever.invoke(item['question'])
                retrieved_texts = [doc.page_content for doc in retrieved_docs]
                        
                reference_context = chunker.split_text(item['context'])
                result = metric.single_turn_score(
                    SingleTurnSample(
                        retrieved_contexts=retrieved_texts,
                        reference_contexts=reference_context,
                    )
                )
                results.append(result)
                progress_bar.update(1)
            progress_bar.close()
            expp[metric.name] = np.mean(results)

        exp_vs.append(expp)
    with open("abl_rag_vs.json", "w") as f:
        json.dump(exp_vs, f, indent=4)


