from langchain_huggingface import HuggingFaceEmbeddings
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.document_loaders import TextLoader, PyMuPDFLoader
from langchain_text_splitters import CharacterTextSplitter, RecursiveJsonSplitter
import os
from tqdm import tqdm
import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS, DistanceStrategy
from src.CustomCSVLoader import CustomCSVLoader
from src.CustomYMLLoader import CustomYMLLoader
from src.CustomJSONLoader import CustomJSONLoader
from langchain_core.documents import Document
from nltk.corpus import stopwords
import nltk
from nltk.stem import PorterStemmer

class DocumentStore:
    """
    DocumentStore class
    ===================
    This class wraps the Chroma vector store and provides a simple interface for ingesting documents
    """
    def __init__(self,
                chunk_size:int = 200,
                chunk_overlap:int = 20,
                k:int = 3,
                collection_name:str = None,
                persist_directory:str = None):
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.k = k
        self.index = faiss.IndexFlatL2(768)
        self.docstore = InMemoryDocstore()
        self.embeddings = HuggingFaceEmbeddings()

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.vector_db = None
        nltk.download('stopwords')

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

        documents:list[Document] = []
        progress = tqdm(total=len(os.listdir(directory)), desc="Ingesting documents")
        for root, dirs, files in os.walk(directory):
            for file in files:
                progress.update(1)
                file_docs = self._create_documents(os.path.join(root, file)) 
                for doc in file_docs:   
                    documents.append(doc)
        progress.close()

        self.vector_db:FAISS = FAISS(
            embedding_function=self.embeddings,
            index=self.index,
            docstore=self.docstore,
            index_to_docstore_id={}
        ).from_documents(
            documents,
            embedding=self.embeddings)

    
    def search_mmr(self, query:str, k:int = 4, fetch_k:int = 20, lambda_mult:float = 0.5) -> list[Document]:
        return self.vector_db.max_marginal_relevance_search(
            query=query,
            k=k,
            fetch_k=fetch_k,
            lambda_mult=lambda_mult)
    
    def _remove_stopwords(self, text:str) -> list[str]:
        """
        Remove stopwords from a text
        """
        stop_words = set(stopwords.words('english'))
        return [word for word in text.split() if word not in stop_words]
    
    def _stem_text(self, text:str) -> list[str]:
        """
        Stem text
        """
        pass
    
    def _normalize_text(self, text:str) -> str:
        """
        Normalize text
        """

        # remove stopwords
        tokens = self._remove_stopwords(text)

        # apply stemming
        """ stemmer = PorterStemmer()
        tokens = [stemmer.stem(token) for token in tokens] """

        return " ".join(tokens)

    def _csv_processor(self, file:str) -> list[Document]:
        """
        Process csv file
        """
        return CustomCSVLoader(file).load()
    
    def _json_processor(self, file:str) -> list[Document]:
        """
        Process json file
        """
        json = CustomJSONLoader(file).load()
        chunks =  RecursiveJsonSplitter().split_json(json)
        return [Document(page_content=str(chunk)) for chunk in chunks]
    
    def _yml_processor(self, file:str) -> list[Document]:
        """
        Process yml file
        """
        yml = CustomYMLLoader(file).load()
        chunks = RecursiveJsonSplitter().split_json(yml)
        return [Document(page_content=str(chunk)) for chunk in chunks]

    def _txt_processor(self, file:str) -> list[Document]:
        """
        Process txt file
        """
        # remove stopwords
        docs = TextLoader(file).load()
        return [Document(self._normalize_text(doc.page_content)) for doc in docs]
    
    def _pdf_processor(self, file:str) -> list[Document]:
        """
        Process pdf file
        """
        pdf = PyMuPDFLoader(file).load()
        chunks = SemanticChunker(embeddings=self.embeddings).split_documents(pdf)
        return [Document(page_content=self._normalize_text(chunk.page_content)) for chunk in chunks]

    def _create_documents(self, file:str):
        """
        Process a file creating documents
        """

        processors = {
            ".csv": self._csv_processor,
            ".json": self._json_processor,
            ".yml": self._yml_processor,
            ".txt": self._txt_processor,
            ".pdf": self._pdf_processor
        }
    
        ext = os.path.splitext(file)[-1]

        if ext in processors:
            return processors[ext](file)
        
        return self._txt_processor(file)
    
    def add_text(self, text:str):
        """
        Add text to the document store
        """
        self.vector_db.add_texts([text])