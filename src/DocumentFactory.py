from langchain_core.documents import Document
from langchain_text_splitters import NLTKTextSplitter
from langchain_text_splitters import RecursiveJsonSplitter
import os
from src.CustomCSVLoader import CustomCSVLoader
from src.CustomJSONLoader import CustomJSONLoader
from src.CustomYMLLoader import CustomYMLLoader
import PyPDF2
import os
import nltk
from bs4 import BeautifulSoup
import json

nltk.download('stopwords')
nltk.download('punkt_tab')

config = json.load(open("config/config.json"))

class DocumentFactory:
    @staticmethod
    def from_text(text:str) -> list[Document]:
        tokenizer = NLTKTextSplitter(
            chunk_size=config['CHUNK_SIZE'],
            chunk_overlap=config['CHUNK_SIZE']*config['CHUNK_OVERLAP'],
        )
        texts = tokenizer.split_text(text)
        return [Document(page_content=doc, metadata={"source": "text"}) for doc in texts]
    
    @staticmethod
    def from_text_file(file:str) -> list[Document]:
        assert os.path.exists(file), f"File {file} not found"

        text = open(file, encoding='utf-8').read()
        tokenizer = NLTKTextSplitter(
            chunk_size=config['CHUNK_SIZE'],
            chunk_overlap=config['CHUNK_SIZE']*config['CHUNK_OVERLAP'],
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
            chunk_size=config['CHUNK_SIZE'],
            chunk_overlap=config['CHUNK_SIZE']*config['CHUNK_OVERLAP'],
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
    def from_html(file:str) -> list[Document]:
        assert os.path.exists(file), f"File {file} not found"

        # load html
        html = open(file, encoding='utf-8').read()
        soup = BeautifulSoup(html, 'html.parser')
        for script in soup(["script", "style"]):
            script.extract()
        
        text = soup.get_text(strip=True)
        tokenizer = NLTKTextSplitter(
            chunk_size=config['CHUNK_SIZE'],
            chunk_overlap=config['CHUNK_SIZE']*config['CHUNK_OVERLAP'],
        )
        texts = tokenizer.split_text(text)
        return [Document(page_content=doc, metadata={"source": file}) for doc in texts]
    
    def from_xml(file:str) -> list[Document]:
        assert os.path.exists(file), f"File {file} not found"

        # load xml
        xml = open(file, encoding='utf-8').read()
        soup = BeautifulSoup(xml, 'xml')
        for script in soup(["script", "style"]):
            script.extract()
        
        text = soup.get_text(strip=True)
        tokenizer = NLTKTextSplitter(
            chunk_size=config['CHUNK_SIZE'],
            chunk_overlap=config['CHUNK_SIZE']*config['CHUNK_OVERLAP'],
        )
        texts = tokenizer.split_text(text)
        return [Document(page_content=doc, metadata={"source": file}) for doc in texts]
    
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
            ".pdf": DocumentFactory.from_pdf,
            ".html": DocumentFactory.from_html,
            ".xml": DocumentFactory.from_xml,
        }
    
        ext = os.path.splitext(file)[-1]

        if ext in processors:
            return processors[ext](file)
        else:
            return processors[".txt"](file)
        