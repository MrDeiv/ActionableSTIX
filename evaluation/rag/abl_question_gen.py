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
from langchain_ollama import ChatOllama
from langchain_core.runnables import RunnableSequence
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import ChatPromptTemplate

RANGE = 3

if __name__ == "__main__":

    # Load SQuAD dataset
    ds = load_dataset("wikimedia/wikipedia", "20231101.en", split="train")
    
    model = ChatOllama(model="gpt-oss:20b")
    query = """
    Given the following context: 
    
    {context}

    Generate a question that can be answered by the context.
    """
    chain_qa = RunnableSequence(
        first=ChatPromptTemplate.from_template(query),
        middle=[model],
        last=StrOutputParser()
    )

    dataset = []
    progress_bar = tqdm(total=RANGE, desc="Processing samples")
    for item in ds.select(range(RANGE)):
        id = item['id']
        context = item['text']
        question = chain_qa.invoke({'context': context})

        dataset.append({
            'id': id,
            'context': context,
            'question': question
        })
        progress_bar.update(1)
    progress_bar.close()

    with open("wikidata_subset.json", "w") as f:
        json.dump(dataset, f, indent=4)
    