from langchain_community.vectorstores import Chroma
from langchain_ollama import ChatOllama
from langchain_ollama import OllamaEmbeddings
from langchain.schema.output_parser import StrOutputParser
from langchain_community.document_loaders import PyPDFLoader, CSVLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema.runnable import RunnablePassthrough
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores.utils import filter_complex_metadata
import os
from tqdm import tqdm
import chromadb

class DocumentParser:
    """
    Document parser class
    """
    vector_store = None
    retriever = None
    chain = None

    def __init__(self, model: str, score_threshold: float = 0.5, chunk_size: int = 1024, chunk_overlap: int = 100):
        self.model = ChatOllama(model=model)
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.prompt = PromptTemplate.from_template(
            """
            <s> [INST]
            You are a cybersecurity assistant. You have been asked to answer the following question based on the given context.
            Your ansers must be technical and detailed. No introduction is required.
            [/INST] </s> 
            [INST] Question: {question} 
            Context: {context} 
            Answer: [/INST]
            """
        )

        # initialize the vector store
        if len(os.listdir("vector_store")) > 0:
            client = chromadb.PersistentClient("vector_store")
        else:
            client = None

        self.vector_store = Chroma(
            collection_name="vector_store",
            persist_directory= "vector_store",
            client=client,
            embedding_function= OllamaEmbeddings(
                model=model,
            ),
        )

        #self.vector_store.persist()
        self.score_threshold = score_threshold

    def ingest(self, directory: str):
        """
        Ingest documents from a directory
        """
        n_docs = len(os.listdir(directory))
        progress = tqdm(total=n_docs, desc="Loading RAG model")
        for file in os.listdir(directory):
            progress.update(1)
            self.add_document(os.path.join(directory, file))

    def _load_document(self, file_path: str):
        """
        Load a document from a file path
        """
        if file_path.endswith(".pdf"):
            loader = PyPDFLoader(file_path)
        elif file_path.endswith(".csv"):
            loader = CSVLoader(file_path)
        else:
            loader = TextLoader(file_path)

        return loader.load()

    def add_document(self, file_path: str):
        """
        Add a Document to the vector store
        """
        # check if the file exists
        assert os.path.exists(file_path), f"File {file_path} does not exist."

        # load the document
        raw_docs = self._load_document(file_path)

        # split the document into chunks
        chunks = self.text_splitter.split_documents(raw_docs)
        chunks = filter_complex_metadata(chunks)

        # add the chunks to the vector store
        self.vector_store.add_documents(chunks)

        # create a retriever
        self.retriever = self.vector_store.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={
                "k": 3,
                "score_threshold": self.score_threshold,
            },
        )

        self.chain = ({"context": self.retriever, "question": RunnablePassthrough()}
                      | self.prompt
                      | self.model
                      | StrOutputParser())
        
        #self.vector_store.persist()

    def ask(self, query: str):
        if not self.chain:
            return "Please, add a PDF document first."

        return self.chain.invoke(query)

    def clear(self):
        self.vector_store = None
        self.retriever = None
        self.chain = None