from langchain_huggingface import HuggingFaceEmbeddings
import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

class DocumentStore:
    """
    DocumentStore class
    ===================
    This class wraps the Chroma vector store and provides a simple interface for ingesting documents
    """
    def __init__(self,
                chunk_size:int = 400,
                chunk_overlap:int = 40,
                k:int = 3,
                collection_name:str = None,
                persist_directory:str = None):
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.k = k
        self.index = faiss.IndexFlatL2(1024)
        self.docstore = InMemoryDocstore()
        self.embeddings = HuggingFaceEmbeddings()

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.vector_db = None

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
        
        self.retriever = self.vector_db.as_retriever()
        
    def add_documents(self, documents:list[Document]) -> list[str]:
        return self.vector_db.add_documents(documents)
    
    def search_mmr(self, query:str, k:int = 4, fetch_k:int = 20, lambda_mult:float = 0.5) -> list[Document]:
        return self.vector_db.max_marginal_relevance_search(
            query=query,
            k=k,
            fetch_k=fetch_k,
            lambda_mult=lambda_mult)
    
    def similarity_search(self, query:str, k:int = 4, fetch_k:int=20) -> list[Document]:
        return self.vector_db.similarity_search(query, k, fetch_k=fetch_k)