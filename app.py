import json, os, dotenv, time
from tqdm import tqdm

from src.STIXParser import STIXParser
from src.group_attack_patterns import group_attack_patterns

# LangGraph
from langgraph.graph import StateGraph, START, END
from src.agents.State import State
from src.DocumentFactory import DocumentFactory

# LangChain
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain_core.runnables import RunnableLambda, RunnableParallel
from langchain.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_ollama import ChatOllama
from langchain_neo4j import GraphCypherQAChain, Neo4jGraph
from langchain_experimental.graph_transformers import LLMGraphTransformer

from src.stores.DocumentStore import DocumentStore

CONFIG_FILE = "./config.json"

dotenv.load_dotenv()

if __name__ == "__main__":

    """
    Application Setup
    """

    # load config
    config = json.load(open(CONFIG_FILE))
    output = [] # output variable

    # execution graph
    execution_graph = StateGraph(State)

    # load files into documents
    directory = config["DOCUMENTS_DIR"]+"/other"
    documents = []
    n_files = len(os.listdir(directory))

    progress = tqdm(total=n_files, desc="Loading documents")
    for file in os.listdir(directory):
        documents.extend(DocumentFactory.from_file(os.path.join(directory, file)))
        progress.update(1)
    progress.close()

    print("Documents loaded:", len(documents))

    """
    Docstore Setup
    """

    # BM25 retriever
    bm25_retriever = BM25Retriever.from_documents(documents)
    bm25_retriever.k = config['BM25_k']

    # Vector store
    docstore = DocumentStore()
    docstore.ingest(documents)

    # Ensemble retriever
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, docstore.retriever],
        weights=[0.4, 0.6]
    )

    # Neo4j
    graph = Neo4jGraph(
        url=os.getenv("NEO4J_URI"),
        username=os.getenv("NEO4J_USER"),
        password=os.getenv("NEO4J_PASSWORD"),
    )
    graph.query("MATCH (n) DETACH DELETE n")

    """
    Documents summarisation
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

    retriever_query = """
    You MUST describe how the malware works.
    """

    llm = ChatOllama(model="llama3.1:8b", temperature=0, verbose=True)

    # build summary
    prompt = ChatPromptTemplate.from_template(summary_query)
    summary_chain = create_stuff_documents_chain(llm, prompt)
    summarized_overview = summary_chain.invoke({
        "context": ensemble_retriever.invoke(retriever_query)
    })

    llm_transformer = LLMGraphTransformer(
        llm=llm, 
        ignore_tool_usage=True)
    
    # prepare to add the summarised overview to the graph
    out_docs = DocumentFactory.from_text(summarized_overview)
    
    # convert the documents to graph documents and add them to the graph
    print("Converting documents to graph documents")
    start = time.time()
    graph_documents = llm_transformer.convert_to_graph_documents(out_docs)
    graph.add_graph_documents(graph_documents)
    end = time.time()
    print("Time taken", end - start, "seconds")

    """
    STIX Parsing
    """
    # load the STIX file
    stix_parser = STIXParser()
    stix_parser.parse(config['STIX_FILE'])

    # extract the attack patterns, malware, and indicators
    attack_patterns = stix_parser.extract_attack_patterns()
    malware_patterns = stix_parser.extract_malware()
    indicators_patterns = stix_parser.extract_indicators()

    attack_patterns_used = stix_parser.get_attack_pattern_used()

    # group the attack patterns
    mitre_tactics = json.loads(open("mitre-tactics.json").read())
    grouped_patterns = group_attack_patterns(mitre_tactics, attack_patterns_used)

    # this are all the hashes mentioned in the malware's iocs, i.e., related files
    #hashes_from_indicators = get_hashes(indicators_patterns) 

    execution_graph.add_node() # ingest
    execution_graph.add_node() # stix
    execution_graph.add_node() # compute attack steps


    # save output
    with open(os.path.join(config["OUTPUT_DIR"], config["OUTPUT_FILE"]), "w") as f:
        json.dump(output, f)