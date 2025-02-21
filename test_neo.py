from typing import List
from langchain.chains import GraphCypherQAChain
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

    # model
    model_name = "microsoft/Phi-3.5-mini-instruct"
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    hf_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer, device="cpu", trust_remote_code=True, temperature=1, do_sample=True)
    """ llm = HuggingFaceEndpoint(
        repo_id=model_name,
        huggingfacehub_api_token=os.getenv("HUGGINGFACE_API_TOKEN"),
        task="text-generation",
    ) """
    #llm = HuggingFacePipeline(pipeline=hf_pipeline, verbose=True)
    #llm = ChatHuggingFace(llm=llm, verbose=True)
    llm = ChatGroq(temperature=0, model_name="llama-3.1-8b-instant",  groq_api_key=os.getenv("GROQ_API_KEY"))
    #llm = ChatOllama(model="llama3.1:8b", temperature=0.1)
    
    llm_transformer = LLMGraphTransformer(llm=llm, ignore_tool_usage=True, allowed_nodes=ALLOWED_NODES, allowed_relationships=ALLOWED_RELATIONS)

    file = "./documents/other/NCSC-MAR-Goofy-Guineapig.pdf"
    documents = DocumentFactory.from_file(file)

    text = """
    Marie Curie, born in 1867, was a Polish and naturalised-French physicist and chemist who conducted pioneering research on radioactivity.
    She was the first woman to win a Nobel Prize, the first person to win a Nobel Prize twice, and the only person to win a Nobel Prize in two scientific fields.
    Her husband, Pierre Curie, was a co-winner of her first Nobel Prize, making them the first-ever married couple to win the Nobel Prize and launching the Curie family legacy of five Nobel Prizes.
    She was, in 1906, the first woman to become a professor at the University of Paris.
    """
    text = """
    Goofy Guineapig is a malicious DLL which is loaded by a legitimate signed executable and maintains persistence using a Windows service. Many defence evasion techniques are implemented, including checking the properties of the infected machine, as well as the running processes and system time checks for any indication the process is running in an automated analysis environment. More information on these checks can be found in the ‘Functionality (Defence Evasion)’ section of this report.
Once loaded Goofy Guineapig can be tasked to collect information about the infected machine or run additional plugins either as part of the current process, or by process hollowing dllhost.exe to execute the plugin. Detailed information about the tasking can be found in the ‘Functionality (Tasking)’ section of this report.
Command and control communications are configured to occur over HTTPS using GET and POST requests to static[.]tcplog[.]com. Full details on C2 are in the ‘Functionality (Communications)’ section of this report.
Loading process
The malicious DLL Goopdate.dll is loaded by the legitimate signed executable file GoogleUpdate.exe. These files are both bundled in a UPX packed NSIS installer which is a trojanised Firefox installer.
The first time the binary is executed, the Goopdate.dll DLL checks if it is running from the location:
C:\ProgramData\GoogleUpdate
If it is not, a service is started for persistence as described in the ‘Functionality (Persistence)’ section of this report.
The initial Goopdate.dll execution writes some commands to a batch file, then creates a hidden process, which calls the batch file via the command line:
cmd /c call C:\<path>\tmp.bat
The first command sets echo to be off; the second command is:
choice /t %d /d y /n >nul
The format string ‘%d’ is never replaced with a numeric value, therefore when executed this command will error, the script will continue on to run the subsequent commands. This was likely intended to provide a delay mechanism between execution and deletion.
The batch script will then delete the files from the original file path of GoogleUpdate.exe and Goopdate.dll, before re-starting the GoogleUpdate.exe process from the ProgramData directory. The final command in the batch script deletes itself.
As a result, the initial directory to which the files were downloaded will only contain the files the recipient likely intended to download, relating to Firefox installation. The malicious files will only be present in the ProgramData directory, which is a hidden directory by default so could be overlooked.
"""
    #chunks = NLTKTextSplitter(chunk_size=400, chunk_overlap=400*0.3).split_text(text)
    #documents = [Document(page_content=chunk) for chunk in chunks]

    print("Converting documents to graph documents")
    print("Started at", time.asctime())
    start = time.time()
    graph_documents = []
    
    batches = [documents[i:i+5] for i in range(0, len(documents), 5)]
    progress = tqdm(total=len(batches))
    for batch in batches:
        progress.update(1)
        graph_documents.extend(llm_transformer.convert_to_graph_documents(batch))
        time.sleep(90)
    progress.close()


    print(f"Conversion took {time.time()-start:.2f} seconds")

    for doc in graph_documents:
        print("Number of nodes:", len(doc.nodes))
        print("Number of edges:", len(doc.relationships))

    print("Adding graph documents to graph")
    graph.add_graph_documents(graph_documents)