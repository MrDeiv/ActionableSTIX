import json, os, dotenv, time, asyncio, re
from tqdm import tqdm
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk import word_tokenize
from sentence_transformers import SentenceTransformer

from src.STIXParser import STIXParser
from src.group_attack_patterns import group_attack_patterns

from transformers import pipeline

# LangGraph
from langgraph.graph import StateGraph, START, END
from src.agents.State import State
from src.DocumentFactory import DocumentFactory
from src.QAModel import QAModel

# LangChain
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnableSequence, Runnable, RunnablePassthrough
from langchain.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_ollama import ChatOllama
from langchain_neo4j import GraphCypherQAChain, Neo4jGraph
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain.schema import Document
from langchain.output_parsers import BooleanOutputParser
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline

from src.stores.DocumentStore import DocumentStore

CONFIG_FILE = "./config.json"

def get_hashes(indicators: list[dict]) -> list[str]:
    """
    Extracts the hashes from the indicators
    """
    hashes:list[str] = []
    
    # filter the indicators that have pattern_type = stix
    stix_indicators = list(filter(lambda x: x['pattern_type'] == 'stix', indicators))
    for ioc in stix_indicators:
        if "file:hashes" in ioc['pattern']:
            h = re.search(r'[a-f0-9]{32,}', ioc['pattern']).group()
            hashes.append(h)
    return hashes

dotenv.load_dotenv()

nltk.download('stopwords')
nltk.download('wordnet')

async def main():

    """
    Application Setup
    """

    # load config
    config = json.load(open(CONFIG_FILE))
    output = [] # output variable

    # load files into documents
    directory = os.path.join(config["DOCUMENTS_DIR"], "other")
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

    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)

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
    hashes_from_indicators = get_hashes(indicators_patterns) 
    #print(hashes_from_indicators)

    """
    Start computing pre-conditions for initial state
    """
    state = {}

    query = """
    You MUST state if the malware is capable of communition with external server.
    The context is: {context}
    """

    ollama_llm = ChatOllama(model="llama3.1:8b", temperature=0, verbose=True)

    ###################################################
    

    chain_summary = RunnableSequence(
        first=ChatPromptTemplate.from_template(query),
        middle=[llm],
        last=StrOutputParser()
    )

    retriever_query = "Does the malware communicate with an external server or necessitate connectivity to work?"
    connectivity_summary = await chain_summary.ainvoke({"context": ensemble_retriever.invoke(retriever_query)})    

    query = """
    Given the following context: {context}

    You MUST state if the malware is capable of communition with external server: write YES or NO.
    """

    chain = RunnableSequence(
        first=ChatPromptTemplate.from_template(query),
        middle=[ollama_llm],
        last=BooleanOutputParser()
    )

    connectivity_out = await chain.ainvoke({"context": connectivity_summary})
    print("[+] Connectivity computed")

    ###################################################

    query = """
    You MUST state the operative system necessary to run the malware.
    If the answer is not directly stated in the text, you MUST infer the answer.

    {context}
    """

    chain_os = RunnableSequence(
        first=ChatPromptTemplate.from_template(query),
        middle=[ollama_llm],
        last=StrOutputParser()
    )
    
    retriever_query = "What operating system is required to run the malware?"
    os_summary = await chain_os.ainvoke({"context": ensemble_retriever.invoke(retriever_query)})

    query = "Given the following context: {context} You MUST EXTRACT ONLY the operative system necessary to run the malware."
    chain = RunnableSequence(
        first=ChatPromptTemplate.from_template(query),
        middle=[ollama_llm],
        last=StrOutputParser()
    )

    os_output = await chain.ainvoke({"context": os_summary})
    print("[+] OS computed")

    ###################################################

    query = """
    You MUST state which vulnerability the malware exploits.
    If the answer is not directly stated in the text, you MUST infer the answer.
    If the are no vulnerabilities exploited, you MUST determine if the malware disguises itself as a legitimate software.
    
    {context}
    """

    chain_vuln = RunnableSequence(
        first=ChatPromptTemplate.from_template(query),
        middle=[ollama_llm],
        last=StrOutputParser()
    )

    retriever_query = "What vulnerability does the malware exploit? Does the malware disguise itself as a legitimate software?"
    vuln_summary = await chain_vuln.ainvoke({"context": ensemble_retriever.invoke(retriever_query)})
    print("[+] Vulnerability computed")

    state['pre-conditions'] = {
        "connectivity": {
            "value": connectivity_out,
            "summary": connectivity_summary
        },
        "os": {
            "value": os_output,
            "summary": os_summary
        },
        "vulnerability": vuln_summary
    }


    ###################################################


    mitre_techniques = json.loads(open("mitre/mitre-techniques.json").read())
    qa_llm = QAModel(model=config['MODELS']['QA'])

    query_refinement_template = """
    Given this context: {context}.\nYou must state how the action: {action} is performed.
    """

    chain_refinement = RunnableSequence(
        first=ChatPromptTemplate.from_template(query_refinement_template),
        middle=[ollama_llm],
        last=StrOutputParser()
    )

    for tactic in grouped_patterns:
        interesting_techniques = mitre_techniques[tactic]['techniques']

        for action in grouped_patterns[tactic]:
            state['actions'] = []
            action_name = action['name']
            action_description = action['description']

            sentence_transformer = SentenceTransformer('all-MiniLM-L6-v2', token=os.getenv("HF_API_KEY"))
            action_vector = sentence_transformer.encode(action_name + "\n" + action_description)
            
            action_mitre_technique_candidated = []
            for technique in interesting_techniques:
                technique_name = technique['name']
                technique_description = technique['description']

                technique_vector = sentence_transformer.encode(technique_name + "\n" + technique_description)

                similarity = sentence_transformer.similarity(action_vector, technique_vector)
                if similarity > 0.6:
                    action_mitre_technique_candidated.append(technique_name)

            query = """
            Given the following choices: 
            {context}

            You MUST select the most appropriate MITRE Technique for the action called: \n"""+action_name+"""
            and desscription: \n"""+action_description+""".
            You MUST fit the action with the most appropriate MITRE Technique, DO NOT add any additional information.
            You MUST select one choice, DO NOT infer the answer.
            """.format(context=",".join(action_mitre_technique_candidated))

            context = ",".join(action_mitre_technique_candidated) if action_mitre_technique_candidated else "Not provided"
            action_technique_name = qa_llm.invoke(query, context)

            try:
                action_technique_id = list(filter(lambda x: x['name'] == action_technique_name, interesting_techniques))[0]['id']
                action_technique_description = list(filter(lambda x: x['name'] == action_technique_name, interesting_techniques))[0]['description']
            except IndexError:
                action_technique_id = "Not provided"
                action_technique_description = "Not provided"
            
            technique = {
                "id": action_technique_id,
                "name": action_technique_name.capitalize(),
                "description": action_technique_description
            }
            
            query_refinement = "Given this MITRE technique: {context}.\nYou must state how the action: {action}, fit the technique".format(context=action_technique_name, action=action_name)
            docs = ensemble_retriever.invoke(query_refinement)
            refined_description = chain_refinement.invoke({
                "context": docs,
                "action": action_name + " " + action_description
            })

            actions = {
                "name": action_name,
                "description": refined_description,
                "MITRE technique": technique,
                "indicators": []
            }

            state['actions'].append(actions)

            print("Action:", action_name)
            print("ref. description:", refined_description)
            print("#"*10)

        output.append(state)
        state = {}


    # save output
    with open(os.path.join(config["OUTPUT_DIR"], config["OUTPUT_FILE"]), "w") as f:
        json.dump(output, f)


if __name__ == "__main__":
    asyncio.run(main())