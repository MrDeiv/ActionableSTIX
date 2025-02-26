from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import NLTKTextSplitter
from langchain_community.document_loaders import TextLoader, UnstructuredHTMLLoader
from langchain_text_splitters import CharacterTextSplitter, RecursiveJsonSplitter, RecursiveCharacterTextSplitter
import os
from src.CustomCSVLoader import CustomCSVLoader
from src.CustomJSONLoader import CustomJSONLoader
from src.CustomYMLLoader import CustomYMLLoader
import PyPDF2
import re
import os
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')
nltk.download('punkt_tab')

class DocumentFactory:
    @staticmethod
    def from_text(text:str) -> list[Document]:
        tokenizer = NLTKTextSplitter(
            chunk_size=400,
            chunk_overlap=400*0.3,
        )
        texts = tokenizer.split_text(text)
        return [Document(page_content=doc, metadata={"source": "text"}) for doc in texts]
    
    @staticmethod
    def from_text_file(file:str) -> list[Document]:
        assert os.path.exists(file), f"File {file} not found"

        text = open(file, encoding='utf-8').read()
        tokenizer = NLTKTextSplitter(
            chunk_size=400,
            chunk_overlap=400*0.3,
        )
        texts = tokenizer.split_text(text)
        return [Document(page_content=doc, metadata={"source": file}) for doc in texts]
    
    @staticmethod
    def from_pdf(file:str) -> list[Document]:
        assert os.path.exists(file), f"File {file} not found"

        # load pdf
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text()

        tokenizer = NLTKTextSplitter(
            chunk_size=400,
            chunk_overlap=400*0.3,
        )
        texts = tokenizer.split_text(text)
        return [Document(page_content=doc, metadata={"source": file}) for doc in texts]
    
    @staticmethod
    def from_json(file:str) -> list[Document]:
        assert os.path.exists(file), f"File {file} not found"

        # load json
        json = CustomJSONLoader(file).load()
        chunks = RecursiveJsonSplitter().create_documents([json], metadatas=[{"source": file}])
        return chunks
    
    @staticmethod
    def from_csv(file:str) -> list[Document]:
        assert os.path.exists(file), f"File {file} not found"

        # load csv
        csv = CustomCSVLoader(file).load()
        return [Document(page_content=str(chunk), metadata={"source": file}) for chunk in csv]
    
    @staticmethod
    def from_yml(file:str) -> list[Document]:
        assert os.path.exists(file), f"File {file} not found"

        # load yml
        yml = CustomYMLLoader(file).load()
        chunks = RecursiveJsonSplitter().create_documents([yml], metadatas=[{"source": file}])
        return chunks
    
    @staticmethod
    def from_file(file:str) -> list[Document]:
        """
        Process a file creating documents
        """

        processors = {
            ".csv": DocumentFactory.from_csv,
            ".json": DocumentFactory.from_json,
            ".yml": DocumentFactory.from_yml,
            ".txt": DocumentFactory.from_text_file,
            ".pdf": DocumentFactory.from_pdf
        }
    
        ext = os.path.splitext(file)[-1]

        if ext in processors:
            return processors[ext](file)
        else:
            return processors[".txt"](file)
        