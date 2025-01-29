import os
import time
import json
from typing import List
import torch
import logging
from transformers import AutoTokenizer, pipeline

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import ChatHuggingFace
from langchain.docstore.document import Document

import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain.schema.runnable import RunnablePassthrough

import pandas as pd
import json
import base64
import io

from langchain_community.chat_models import ChatOllama
from langchain_ollama import ChatOllama
from langchain_ollama import OllamaEmbeddings
from langchain.prompts import PromptTemplate
from langchain.vectorstores.utils import filter_complex_metadata
from langchain.schema.output_parser import StrOutputParser

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

    file_to_load = "documents/21-00021921.pdf"
    ds = DocumentStore()
    


    # toc
    toc = time.time()
    print(f"Time Elapsed: {toc - tic:.2f} s")

