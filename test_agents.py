from langchain_community.tools import DuckDuckGoSearchResults
from tqdm import tqdm
from src.DocumentFactory import DocumentFactory
from src.stores.DocumentStore import DocumentStore
from src.TextGenerationModel import TextGenerationModel
import os
from src.build_context import build_context
from src.tools.RetrieverInterface import RetrieverTool
from smolagents import CodeAgent, HfApiModel
from transformers import (AutoModelForCausalLM, AutoTokenizer, TextGenerationPipeline, pipeline)

if __name__ == '__main__':

    documents_dir = "./documents/other" # other documents (not from anyrun or vt)

    documents = []
    progress = tqdm(total=len(os.listdir(documents_dir)), desc="Ingesting Documents")
    for file in os.listdir(documents_dir):
        documents.extend(DocumentFactory.from_file(os.path.join(documents_dir, file)))
        progress.update(1)
    progress.close()

    docstore = DocumentStore()
    docstore.ingest(documents)

    # Create the RetrieverAgent
    """ model = "microsoft/Phi-3.5-mini-instruct"
    question = "Given the context, must the machine be connected to internet to complete the malware attack?"
    retriever = RetrieverTool(model, docstore=docstore)
    docs = retriever.run(question)
    print(docs) """


    