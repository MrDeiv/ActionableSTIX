from langchain_ollama import OllamaEmbeddings
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.document_loaders import TextLoader, CSVLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
from tqdm import tqdm
import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS, DistanceStrategy

class DocumentStore:
    """
    DocumentStore class
    ===================
    This class wraps the Chroma vector store and provides a simple interface for ingesting documents
    """
    def __init__(self,
                model:str,
                collection_name:str = None,
                persist_directory:str = None):
        self.collection_name = collection_name
        self.persist_directory = persist_directory

        self.embeddings = OllamaEmbeddings(model=model)

        self.index = faiss.IndexFlatL2(len(self.embeddings.embed_query("hello world")))
        self.vector_store = FAISS(
            embedding_function=self.embeddings,
            index=self.index,
            docstore=InMemoryDocstore(),
            index_to_docstore_id = {}
        )

        self.retriever = self.vector_store.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={
                'k': 3,
                "score_threshold": 0.5,
            }
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
    
    def ingest_directory(self, directory:str):
        """
        Ingests a directory of files into the vector database
        """
        assert os.path.isdir(directory), f"{directory} is not a directory"

        progress = tqdm(total=len(os.listdir(directory)), desc="Ingesting documents")
        for root, dirs, files in os.walk(directory):
            for file in files:
                print(f"Ingesting {file}")
                progress.update(1)
                self.ingest_document(os.path.join(root, file))
        progress.close()

    def ingest_document(self, file:str):
        """
        Load file and ingest into vector database
        """
        loader, splitter = self._get_processors(file)
        document = loader.load()
        if splitter is not None:
            chunks = splitter.split_documents(document)
        else:
            chunks = document
        
        self.vector_store.add_documents(chunks)

    def _get_processors(self, file:str):
        """
        Get the appropriate document loader and splitter for the file
        """

        if file.endswith(".csv"):
            return CSVLoader(file), None
        
        if file.endswith(".txt"):
            return TextLoader(file), RecursiveCharacterTextSplitter(
                chunk_size=100, 
                chunk_overlap=20,
                length_function=len
            )
        
        if file.endswith(".pdf"):
            return PyPDFLoader(file), SemanticChunker(
                embeddings=self.embeddings,
                breakpoint_threshold_type="standard_deviation"
            )
        
        return TextLoader(file), None

