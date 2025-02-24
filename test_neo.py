from typing import List
from langchain_neo4j import GraphCypherQAChain
from langchain_community.graphs import Neo4jGraph
import dotenv
import os
from langchain_experimental.graph_transformers import LLMGraphTransformer
from transformers import (AutoModelForCausalLM, AutoTokenizer, pipeline)
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEndpoint, HuggingFacePipeline, ChatHuggingFace
from tqdm import tqdm
from src.DocumentFactory import DocumentFactory
import time
import warnings
from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama
from langchain_community.graphs.graph_document import GraphDocument
from langchain_text_splitters import NLTKTextSplitter
import nltk

warnings.filterwarnings("ignore")

ALLOWED_NODES = ["Command", "DLL", "Directory", "Domain", "Executable", "File", "Installer", "Malware", "Process", "Protocol", "Communication", "Request", "Script", "Service", "Technique", "Tactic", "Tool", "Vulnerability"]

ALLOWED_RELATIONS = ["BUNDLED_WITH", "CALLS_COMMAND", "CAN_EXECUTE_PLUGIN", "CHECKS_LOCATION", "CREATES_FILE", "CREATES_PROCESS", "DELETES", "ERRORS_ON_EXECUTION", "EXECUTES", "HAS_PROPERTY", "HOLLOWED_BY", "LOADS_DLL", "PERSISTS_AS_SERVICE", "RESTARTS", "SENDS_REQUEST", "STARTS_SERVICE", "TASKS", "USES_PROTOCOL", "IMPLEMENTED_DEFENCE_EVASION_TECHNIQUE", "IS_TROJANISED", "EXPLOITS", "HAS_VULNERABILITY", "HAS_TACTIC", "HAS_TECHNIQUE", "HAS_TOOL", "HAS_MALWARE", "HAS_DOMAIN", "HAS_FILE", "HAS_DIRECTORY", "HAS_EXECUTABLE", "HAS_PROCESS", "HAS_SCRIPT", "HAS_COMMUNICATION", "HAS_REQUEST", "LOADS", "MAINTAINS_PERSISTENCE", "CHECKS_SYSTEM_TIME", "CHECKS_RUNNING_PROCESSES", "CHECKS_MACHINE_PROPERTIES", "CHECKS_ENVIRONMENT", "CHECKS_FOR_ANALYSIS_ENVIRONMENT", "CHECKS_FOR_SANDBOX", "CHECKS_FOR_DEBUGGER", "DOWNLOADS", "EXECUTES"]

if __name__ == "__main__":

    assert dotenv.load_dotenv(), "No .env file found"

    # graph database
    graph = Neo4jGraph(
        url=os.getenv("NEO4J_URI"),
        username=os.getenv("NEO4J_USER"),
        password=os.getenv("NEO4J_PASSWORD"),
    )

    # clear graph
    graph.query("MATCH (n) DETACH DELETE n")

    #llm = ChatGroq(temperature=0, model_name="llama-3.1-8b-instant",  groq_api_key=os.getenv("GROQ_API_KEY"))
    llm = ChatOllama(model="llama3.1:8b", temperature=0, verbose=True)
    
    llm_transformer = LLMGraphTransformer(llm=llm, ignore_tool_usage=True)

    #file = "./documents/other/NCSC-MAR-Goofy-Guineapig.pdf"
    #documents = DocumentFactory.from_file(file)

    text = open("summary.txt", "r").read()
    
    chunk_size = 400
    chunks = NLTKTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_size*0.3).split_text(text)
    documents = [Document(page_content=chunk) for chunk in chunks]

    print("Converting documents to graph documents")
    print("Started at", time.asctime())
    start = time.time()
    
    graph_documents = llm_transformer.convert_to_graph_documents(documents)
    
    end = time.time() 
    print("Finished at", time.asctime())
    print("Time taken", end - start, "seconds")

    assert len(graph_documents) > 0, "No graph documents were created"
    assert all([doc.nodes for doc in graph_documents]), "Some graph documents have no nodes"
    assert all([doc.relationships for doc in graph_documents]), "Some graph documents have no relations"

    print("Adding graph documents to graph")
    graph.add_graph_documents(graph_documents)

    #exit()

    chain = GraphCypherQAChain.from_llm(
        ChatOllama(model="llama3.1:8b", temperature=0, verbose=True),
        graph = graph,
        verbose = True,
        allow_dangerous_requests=True,
        use_function_response=True,
        validate_cypher=True,
    )

    query = "Given the context, you MUST state which prize the Curie won. DO NOT add any additional information."
    response = chain.invoke({"query": query})['result']

    print(response)