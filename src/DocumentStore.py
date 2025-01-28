from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import UnstructuredMarkdownLoader, PyPDFLoader, CSVLoader, UnstructuredHTMLLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.utils import filter_complex_metadata
import os
from tqdm import tqdm

class DocumentStore:
    """
    DocumentStore class
    ===================
    This class wraps the Chroma vector store and provides a simple interface for ingesting documents
    """
    def __init__(self,
                model_name:str,
                collection_name:str = None,
                persist_directory:str = None,
                chunk_size:int = 1024,
                chunk_overlap:int = 100):
        self.model_name = model_name
        self.collection_name = collection_name
        self.persist_directory = persist_directory

        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.model_name
        )

        self.vector_db = Chroma(
            collection_name=self.collection_name,
            persist_directory=self.persist_directory,
            embedding_function=self.embeddings
        )

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

    @property
    def embeddings(self):
        return self._embeddings
    
    @property
    def vector_db(self):
        return self._vector_db
    
    @embeddings.setter
    def embeddings(self, value):
        self._embeddings = value
    
    @vector_db.setter
    def vector_db(self, value):
        self._vector_db = value
    
    def ingest(self, directory:str):
        """
        Ingests a directory of files into the vector database
        """
        assert os.path.isdir(directory), f"{directory} is not a directory"

        progress = tqdm(total=len(os.listdir(directory)), desc="Ingesting documents...")
        for root, dirs, files in os.walk(directory):
            for file in files:
                progress.update(1)
                self._ingest_document(os.path.join(root, file))
        progress.close()

    def _ingest_document(self, file:str):
        """
        Load file and ingest into vector database
        """
        loader = self._get_loader(file)
        chunks = loader(file).load_and_split(
            text_splitter=self.splitter
        )
        self.vector_db.add_documents(filter_complex_metadata(chunks))

    def _get_loader(self, file:str):
        """
        Returns the appropriate loader for the file
        """
        if file.endswith(".md"):
            return UnstructuredMarkdownLoader
        elif file.endswith(".pdf"):
            return PyPDFLoader
        elif file.endswith(".csv"):
            return CSVLoader
        elif file.endswith(".html"):
            return UnstructuredHTMLLoader
        else:
            return TextLoader

