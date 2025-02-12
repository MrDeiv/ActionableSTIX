import json
import os

from tqdm import tqdm
from src.QAModel import QAModel
from src.SentimentAnalysisModel import SentimentAnalysisModel
from src.SummarizationModel import SummarizationModel
from src.TextGenerationModel import TextGenerationModel
from src.build_context import build_context
from src.group_attack_patterns import group_attack_patterns
from src.DocumentFactory import DocumentFactory
from src.DocumentStore import DocumentStore
from src.STIXParser import STIXParser
import re
from langchain_core.documents import Document

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

CONFIG_FILE = "config.json"

if __name__ == '__main__':
    # load the configuration file
    config = json.loads(open(CONFIG_FILE).read())

    # this is the output variable (list of dictionaries)
    out = []

    """
    1. STIX Loading and Pre-processing
    """
    # load the STIX file
    stix_parser = STIXParser()
    stix_parser.parse(config['STIX_FILE'])

    # extract the attack patterns, malware, and indicators
    attack_patterns = stix_parser.extract_attack_patterns()
    malware_patterns = stix_parser.extract_malware()
    indicators_patterns = stix_parser.extract_indicators()

    # group the attack patterns
    mitre_tactics = json.loads(open("mitre-tactics.json").read())
    grouped_patterns = group_attack_patterns(mitre_tactics, attack_patterns)

    # this are all the hashes mentioned in the malware's iocs, i.e., related files
    hashes_from_indicators = get_hashes(indicators_patterns) 
    
    # ...

    """
    2. Documents Ingestion
    """
    # load the documents
    documents:list[Document] = []
    documents_dir = "./documents/other" # other documents (not from anyrun or vt)

    progress = tqdm(total=len(os.listdir(documents_dir)), desc="Ingesting Documents")
    for file in os.listdir(documents_dir):
        documents.extend(DocumentFactory.from_file(os.path.join(documents_dir, file)))
        progress.update(1)
    progress.close()

    """
    3. Document Store Ingestion
    """
    docstore = DocumentStore()
    docstore.ingest(documents)

    """
    4. Enrichment Process
    """
    # define models
    qa_model = QAModel(config['MODELS']['QA'])
    text_generation_model = TextGenerationModel(config['MODELS']['TEXT_GENERATION'])
    summary_model = SummarizationModel(config['MODELS']['SUMMARIZATION'])
    sentiment_model = SentimentAnalysisModel(config['MODELS']['TEXT_GENERATION'])

    # connectivity requirements
    question = "Given the context, must the machine be connected to internet to complete the malware attack?"
    context = docstore.similarity_search(question, k=3, fetch_k=10)
    context = build_context(context)

    prompt_template = f"""
    Given the following context, you must answer the question with considering only
    the information provided in the context.
    Do not add any consideration or information that is not present in the context.
    Do not add ny introduction.
    If the answer is not directly present in the context, you can infer it.
    If you infer the answer, please provide the reasoning.
    The context is as follows:
    {context}
    """
    prompt = prompt_template.format(context=context)

    connection_required_summary = text_generation_model.invoke(question, prompt)

    # map to True or False
    query =f"""
    Given the statement:
    {connection_required_summary}

    Map it to True or False. Do not add any additional information.
    """
    connectivity_required = sentiment_model.invoke(query)
    print("[+] Connectivity Required:", connectivity_required)

    # operating system requirements
    question = """
    Given the context, for which operating system is the malware designed?
    Is it Windows, Linux, MacOS, or other?
    """
    context = docstore.similarity_search(question, k=3, fetch_k=10)
    context = build_context(context)
    prompt = prompt_template.format(context=context)
    os_required_summary = text_generation_model.invoke(question, prompt)
    operating_system = qa_model.invoke(question, os_required_summary)
    print("[+] Operating System:", operating_system)    
    
    # vulnerable software requirements
    question = """
    Given the context, is the malware designed to exploit a specific software vulnerability?
    If yes, which software is vulnerable?
    Otherwise, is it designed to mimic a legitimate software?
    If yes, which software is mimicked?
    """
    context = docstore.similarity_search(question, k=3, fetch_k=10)
    context = build_context(context)
    prompt = prompt_template.format(context=context)
    vulnerable_software_summary = text_generation_model.invoke(question, prompt)
    question = """
    Given the context, extract only the name of the software that is vulnerable or mimicked by the malware.
    Do not add any additional information or text.
    Provide the names as a comma-separated list.
    """
    software = text_generation_model.invoke(question, vulnerable_software_summary)
    
    print("[+] Software:", software)

    stage = {}
    stage['pre-conditions'] = {
        "is_connectivity_required": {
            "short": connectivity_required,
            "summary": connection_required_summary
        },
        "os": {
            "short": operating_system,
            "summary": os_required_summary
        },
        "target_software": {
            "short": software,
            "summary": vulnerable_software_summary,
        }  
    }
    
    for group in grouped_patterns.keys():
        stage['actions'] = []
        exit()