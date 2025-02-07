from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_experimental.text_splitter import SemanticChunker
from langchain_text_splitters import NLTKTextSplitter
from langchain_community.document_loaders import TextLoader
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
    def _remove_yara_rules(text):
        yara_pattern = r'(?s)\brule\b.*?\{.*?\n\}'
        cleaned_text = re.sub(yara_pattern, '', text)
        cleaned_text = re.sub(r'\n\s*\n', '\n\n', cleaned_text)
        return cleaned_text.strip()
    
    @staticmethod
    def _remove_stopwords(text:str) -> str:
        stop_words = set(stopwords.words('english'))
        words = text.split()
        cleaned_text = " ".join([word for word in words if word.lower() not in stop_words])
        return cleaned_text

    @staticmethod
    def from_text(file:str) -> list[Document]:
        assert os.path.exists(file), f"File {file} not found"

        # load text
        text_documents = TextLoader(file).load()
        chunks = CharacterTextSplitter().split_documents(text_documents)
        return chunks
    
    @staticmethod
    def from_pdf(file:str) -> list[Document]:
        assert os.path.exists(file), f"File {file} not found"

        # load pdf
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text()

        tokenizer = NLTKTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
        )
        texts = tokenizer.split_text(text)
        return [Document(page_content=doc) for doc in texts]
    
    @staticmethod
    def from_json(file:str) -> list[Document]:
        assert os.path.exists(file), f"File {file} not found"

        # load json
        json = CustomJSONLoader(file).load()
        chunks = RecursiveJsonSplitter().create_documents([json])
        return chunks
    
    @staticmethod
    def from_csv(file:str) -> list[Document]:
        assert os.path.exists(file), f"File {file} not found"

        # load csv
        csv = CustomCSVLoader(file).load()
        return [Document(page_content=str(chunk)) for chunk in csv]
    
    @staticmethod
    def from_yml(file:str) -> list[Document]:
        assert os.path.exists(file), f"File {file} not found"

        # load yml
        yml = CustomYMLLoader(file).load()
        chunks = RecursiveJsonSplitter().create_documents([yml])
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
            ".txt": DocumentFactory.from_text,
            ".pdf": DocumentFactory.from_pdf
        }
    
        ext = os.path.splitext(file)[-1]

        if ext in processors:
            return processors[ext](file)
        else:
            return processors[".txt"](file)
        
    @staticmethod
    def from_content(content:str) -> list[Document]:
        return [Document(page_content=content)]