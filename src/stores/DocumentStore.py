from langchain_ollama import OllamaEmbeddings
import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

class DocumentStore():
    """
    DocumentStore class
    ===================
    This class wraps the Chroma vector store and provides a simple interface for ingesting documents
    """
    def __init__(self, k:int = 3):
        self.index = faiss.IndexFlatL2(1024)
        self.docstore = InMemoryDocstore()
        self.embeddings = OllamaEmbeddings(model="llama3.1:8b")
        self.vector_db = None
        self.k = k

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
    
    def ingest(self, documents:list[Document]):
        """
        Ingests documents into the document store
        """
        self.vector_db:FAISS = FAISS(
            embedding_function=self.embeddings,
            index=self.index,
            docstore=self.docstore,
            index_to_docstore_id={}
        ).from_documents(
            documents,
            embedding=self.embeddings)
        
        self.retriever = self.vector_db.as_retriever(
            search_type="similarity",
            k=self.k,
        )
        
    def add_documents(self, documents:list[Document]) -> list[str]:
        return self.vector_db.add_documents(documents)