import time
import json
from transformers import AutoTokenizer
from tqdm import tqdm

from langchain_text_splitters import RecursiveCharacterTextSplitter, CharacterTextSplitter, SentenceTransformersTokenTextSplitter, NLTKTextSplitter
from langchain_experimental.text_splitter import SemanticChunker as LangchainSemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings

from chonkie import TokenChunker, WordChunker, SentenceChunker, RecursiveChunker, RecursiveRules, SemanticChunker, SDPMChunker, LateChunker, Chunk

CHUNK_SIZE = 400
CHUNK_OVERLAP = 0.3

if __name__ == '__main__':
    state_of_union_txt = open("state_of_the_union.txt", "r").read()

    ls_charactertextsplitter = CharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_SIZE*CHUNK_OVERLAP
    )

    ls_recursivecharacter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_SIZE*CHUNK_OVERLAP
    )

    ls_nltk = NLTKTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_SIZE*CHUNK_OVERLAP
    )

    # test the splitters
    chunks = ls_nltk.create_documents([state_of_union_txt])

    print(chunks[0].page_content)