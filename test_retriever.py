from src.DocumentFactory import DocumentFactory
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain.schema import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain.schema.runnable import Runnable, RunnableSequence, RunnablePassthrough, RunnableLambda
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_neo4j import GraphCypherQAChain, Neo4jGraph
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
import os
import dotenv
import time

dotenv.load_dotenv()

ALLOWED_NODES = ["Command", "DLL", "Directory", "Domain", "Executable", "File", "Installer", "Malware", "Process", "Protocol", "Communication", "Request", "Script", "Service", "Technique", "Tactic", "Tool", "Vulnerability"]
ALLOWED_RELATIONS = ["BUNDLED_WITH", "CALLS_COMMAND", "CAN_EXECUTE_PLUGIN", "CHECKS_LOCATION", "CREATES_FILE", "CREATES_PROCESS", "DELETES", "ERRORS_ON_EXECUTION", "EXECUTES", "HAS_PROPERTY", "HOLLOWED_BY", "LOADS_DLL", "PERSISTS_AS_SERVICE", "RESTARTS", "SENDS_REQUEST", "STARTS_SERVICE", "TASKS", "USES_PROTOCOL", "IMPLEMENTED_DEFENCE_EVASION_TECHNIQUE", "IS_TROJANISED", "EXPLOITS", "HAS_VULNERABILITY", "HAS_TACTIC", "HAS_TECHNIQUE", "HAS_TOOL", "HAS_MALWARE", "HAS_DOMAIN", "HAS_FILE", "HAS_DIRECTORY", "HAS_EXECUTABLE", "HAS_PROCESS", "HAS_SCRIPT", "HAS_COMMUNICATION", "HAS_REQUEST", "LOADS", "MAINTAINS_PERSISTENCE", "CHECKS_SYSTEM_TIME", "CHECKS_RUNNING_PROCESSES", "CHECKS_MACHINE_PROPERTIES", "CHECKS_ENVIRONMENT", "CHECKS_FOR_ANALYSIS_ENVIRONMENT", "CHECKS_FOR_SANDBOX", "CHECKS_FOR_DEBUGGER", "DOWNLOADS", "EXECUTES"]


if __name__ == "__main__":

    # load and chunk the file
    #file = "./documents/other/NCSC-MAR-Goofy-Guineapig.pdf"
    folder = "./documents/other/"

    documents = []
    for file in os.listdir(folder):
        print("File:", file)
        documents.extend(DocumentFactory.from_file(os.path.join(folder, file)))

    # BM25 retriever
    bm25_retriever = BM25Retriever.from_documents(documents)
    bm25_retriever.k = 5 

    # FAISS retriever
    embeddings = OllamaEmbeddings(model="llama3.1:8b")
    faiss_db = FAISS.from_documents(documents, embedding=embeddings)
    faiss_retriever = faiss_db.as_retriever(search_kwargs={"k": 3})

    # Ensemble retriever
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, faiss_retriever],
        weights=[0.4, 0.6]
    )

    # model
    llm = ChatOllama(model="llama3.1:8b", temperature=0, verbose=True)

    retriever_query = """
    You MUST describe how the malware works.
    """

    summary_query = """
    You MUST summarize this content: {context}

    You MUST be precise describing the actions performed by the malware.
    You MUST be precise.
    You MUST use the malware as subject.
    Do NOT add any introduction.
    DO NOT insert any additional information.
    DO NOT use lists.
    DO NOT use any formatting.
    """

    """ summary_llm = HuggingFacePipeline.from_model_id(
        model_id="microsoft/Phi-3.5-mini-instruct",
        task="text-generation",
        pipeline_kwargs={"max_new_tokens": 4096}
    )

    summary_chat = ChatHuggingFace(llm=summary_llm) """

    prompt = ChatPromptTemplate.from_template(summary_query)
    summary_chain = create_stuff_documents_chain(llm, prompt)
    out = summary_chain.invoke({
        "context": ensemble_retriever.invoke(retriever_query)
    })

    # get only the assistant summary
    limit = "<|assistant|>"
    # get last index of limit
    index = out.rfind(limit)
    # get the assistant summary
    summary = out[index + len(limit):].strip().replace("\n", " ")
    print(summary)

    # Neo4j
    graph = Neo4jGraph(
        url=os.getenv("NEO4J_URI"),
        username=os.getenv("NEO4J_USER"),
        password=os.getenv("NEO4J_PASSWORD"),
    )

    # clean the graph
    graph.query("MATCH (n) DETACH DELETE n")

    # set up the transformer
    llm_transformer = LLMGraphTransformer(
        llm=llm, 
        ignore_tool_usage=True)
    
    out_docs = DocumentFactory.from_text(summary)
    
    # convert the documents to graph documents
    print("Converting documents to graph documents")
    start = time.time()
    graph_documents = llm_transformer.convert_to_graph_documents(out_docs)
    end = time.time()

    # add the graph documents to the graph
    print("Adding graph documents to graph")
    start = time.time()
    graph.add_graph_documents(graph_documents)
    end = time.time()
    print("Time taken", end - start, "seconds")

    chain = GraphCypherQAChain.from_llm(
        ChatOllama(model="llama3.1:8b", temperature=0, verbose=True),
        graph = graph,
        verbose = True,
        allow_dangerous_requests=True,
        use_function_response=True,
    )

    query = """
    Does the malware has connectivity requirements?
    """
    response = chain.invoke({"query": query})['result']

    print(response)
