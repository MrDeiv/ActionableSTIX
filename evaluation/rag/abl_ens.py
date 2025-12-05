from ragas import SingleTurnSample
from ragas.metrics import NonLLMContextPrecisionWithReference, NonLLMContextRecall
from datasets import load_dataset, Dataset
from ragas import EvaluationDataset
import numpy as np
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from DocumentStore import DocumentStore
from chunking_evaluation.chunking import ClusterSemanticChunker
from chromadb.utils import embedding_functions
from langchain_core.documents import Document
from tqdm import tqdm
import json

K_BM25 = 50
K = 30
WEIGHTS = [(0.2, 0.8), (0.4, 0.6), (0.5, 0.5), (0.6, 0.4), (0.8, 0.2)]

if __name__ == "__main__":

    # Define metric
    metrics = [
        NonLLMContextPrecisionWithReference(),
        NonLLMContextRecall()
    ]

    default_ef = embedding_functions.DefaultEmbeddingFunction()
    chunker = ClusterSemanticChunker(default_ef, max_chunk_size=200)

    chunks = set()
    references = []
    ds = Dataset.from_json("squad_subset.json")
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
    np.random.seed(42)
    np.random.shuffle(chunks)
    
    # Ensemble retriever
    exp_ens = []
    for weights in WEIGHTS:
        expp = {
            "weights": weights,
        }
        # BM25 retriever
        bm25_retriever = BM25Retriever.from_texts(chunks)
        bm25_retriever.k = K_BM25

        # Vector store
        docstore = DocumentStore(k=K)
        documents = [Document(page_content=chunk) for chunk in chunks]
        docstore.ingest(documents)

        # Ensemble retriever
        ensemble_retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, docstore.retriever],
            weights=weights
        )
        for metric in metrics:
            expp[metric.name] = 0
            print(f"Using metric: {metric.name}")
            results = []
            progress_bar = tqdm(total=len(ds), desc=f"Evaluating {metric.name}")
            for item in ds:
                retrieved_docs = ensemble_retriever.invoke(item['question'])
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
        exp_ens.append(expp)

    with open("abl_rag_ens.json", "w") as f:
        json.dump(exp_ens, f, indent=4)


