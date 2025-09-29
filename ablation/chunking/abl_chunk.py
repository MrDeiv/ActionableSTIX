from chunking_evaluation import BaseChunker, GeneralEvaluation
from chunking_evaluation.chunking import ClusterSemanticChunker
from chromadb.utils import embedding_functions
import os

# Instantiate evaluation
evaluation = GeneralEvaluation()

# Choose embedding function
default_ef = embedding_functions.DefaultEmbeddingFunction()
chunker = ClusterSemanticChunker(default_ef, max_chunk_size=200)

# split
splits = chunker.split_text(
    open("sotu.txt", encoding='utf-8').read().replace("\n", " ")
)

print(f"Number of chunks: {len(splits)}")
print(f"First chunk: {splits[0]}")