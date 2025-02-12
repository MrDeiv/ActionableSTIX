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
from mitreattack.stix20 import MitreAttackData
from sentence_transformers import SentenceTransformer
import nltk
from nltk.corpus import stopwords
import numpy as np
from smolagents import CodeAgent
from src.agents.get_mitre_technique import get_mitre_technique
from transformers import AutoModelForCausalLM
from chonkie import SemanticChunker


nltk.download('stopwords')

def get_mitre_by_id(matrix_file:str, mitre_id: str) -> dict:
    """
    Get the MITRE technique by id
    """
    mitre = MitreAttackData(matrix_file)
    return mitre.get_object_by_attack_id(mitre_id, 'attack-pattern')

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

    attack_patterns_used = stix_parser.get_attack_pattern_used()

    # group the attack patterns
    mitre_tactics = json.loads(open("mitre-tactics.json").read())
    grouped_patterns = group_attack_patterns(mitre_tactics, attack_patterns_used)

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
    #qa_model = QAModel(config['MODELS']['QA'])
    text_generation_model = TextGenerationModel(
        config['MODELS']['TEXT_GENERATION'],
        max_new_tokens=512)
    mitre_model = TextGenerationModel(
        config['MODELS']['TEXT_GENERATION'],
        max_new_tokens=128
    )
    #summary_model = SummarizationModel(config['MODELS']['SUMMARIZATION'])
    #sentiment_model = SentimentAnalysisModel(config['MODELS']['TEXT_GENERATION'])


    stage = {}
    stage['pre-conditions'] = {
        "is_connectivity_required": {
            "short": '',
            "summary": ''
        },
        "os": {
            "short": '',
            "summary": ''
        },
        "target_software": {
            "short": '',
            "summary": '',
        }  
    }

    print("[+] Computed pre-conditions for the malware to run")

    # Start evaluating the actions necessary to change state
    # Download the reports from CTI platforms
    # we should iterate over hashes extracted before, this is just an example
    # to limit API consumption
    target_hash = 'a21dec89611368313e138480b3c94835'
    anyrun_json = json.load(open(f"./documents/anyrun/{target_hash}.json", encoding='utf-8'))
    anyrun_mitre = anyrun_json['mitre']

    # compute embeddings for the mitre techniques in anyrun
    stop_words = set(stopwords.words('english'))
    sent_trasf = SentenceTransformer('all-MiniLM-L6-v2')
    names_embeddings = []
    for tactic in anyrun_mitre:
        name = tactic['name']
        name_normalized = name
        #name_normalized = (' '.join([word for word in name.split() if word not in stop_words])).lower()
        names_embeddings.append(sent_trasf.encode(name_normalized))

    
    anyrun_ioc = json.load(open(f"./documents/anyrun/{target_hash}_ioc.json", encoding='utf-8'))
    chunker = SemanticChunker(
        embedding_model="minishlab/potion-base-8M",  
        threshold=0.5,                               
        chunk_size=512,                              
        min_sentences=1                              
    )

    # load anyrun reports
    for file in os.listdir("./documents/anyrun"):
        if file.endswith(".html"):
            docs = DocumentFactory.from_html(os.path.join("./documents/anyrun", file))
            docstore.add_documents(docs)

    tactics_in_stix = grouped_patterns.keys()
    for tactic in tactics_in_stix:
        actions_per_tactic = grouped_patterns[tactic]
        
        stage_actions = []
        for action in actions_per_tactic:
            action_name = action['name']
            action_description = action['description']

            # link to the MITRE technique -> use semantic similarity
            # the action name should be similar to the MITRE technique name
            
            # remove stopwords
            action_name_normalized = action_name
            #action_name_normalized = (' '.join([word for word in action_name.split() if word not in stop_words])).lower()
            action_name_embedding = sent_trasf.encode(action_name_normalized)

            # compute the similarity
            similarities = []
            for name_embedding in names_embeddings:
                similarities.append(sent_trasf.similarity(action_name_embedding, name_embedding))

            # get the three most similar techniques
            techs = set()
            while len(techs) < 5:
                max_similarity = max(similarities)

                # get the index of the most similar technique
                index = similarities.index(max_similarity)
                similarities.pop(index)

                # get the MITRE technique
                mitre_technique = f"{anyrun_mitre[index]['id']} - {anyrun_mitre[index]['name']}"
                techs.add(mitre_technique)

            question = f"""
            You are given the action name: {action_name}.
            What are the MITRE techniques related to this action between the following ones?
            MITRE techniques proposed:
            {'\n'.join(techs)}

            You MUST choose the most relevant technique between the ones provided.
            DO NOT add any information that is not present in the context or introduction.
            You MUST only state the technique, no more.
            """
            technique:str = mitre_model.invoke(question, "")
            mitre_id, mitre_name = technique.split(" - ")

            # get the MITRE technique description
            #res = agent.run(
            #    f"""
            #    You MUST gather information about the MITRE technique {mitre_id}.
            #    To gather the knowledge, you can use the tool by providing the technique id.
            #    """
            #)

            mitre_description = get_mitre_technique(mitre_id)

            question = f"""
            How the action {action_name} fits into the MITRE technique {mitre_id}?
            """
            context = docstore.similarity_search(question, k=3, fetch_k=10)
            context = build_context(context)

            prompt = f"""
            You will be given the name of the action, its description and the MITRE technique related to it.
            You MUST fit the information from the context into the guidelines provided by MITRE technique.
            DO NOT add any information that is not present in the context or introduction.
            The context is as follows:
            {context}

            The action is: {action_name}
            The description is: {action_description}
            The MITRE technique is: {mitre_id} - {mitre_name}
            The MITRE technique description is: {mitre_description}
            """
            action_description = text_generation_model.invoke(question, prompt)

            question = f"""
            What are the indicators related to the action {action_name}, defined as {action_description}?
            """
            context = docstore.similarity_search(question, k=3, fetch_k=10)
            context = build_context(context)

            prompt = f"""
            You will be given the name of the action and its description.
            You MUST provide the indicators related to the action. These indicators are technical details, AVOID general information.
            You MUST AVOID non-technical information.
            You MUST answer with a list of indicators, comma-separated. If there are no indicators, state that.
            You MUST provide only a bullet list of indicators.
            DO NOT add any information that is not present in the context or introduction.
            The context is as follows:
            {context}
            """
            indicators:str = mitre_model.invoke(question, prompt)
            indicators_list = indicators.strip().split('\n')
            indicators_list = [i[2:] for i in indicators_list]
            
            stage_actions.append({
                "action": action_name,
                "description": action_description,
                "mitre_technique": technique.strip(),
                "mitre_description": mitre_description,
                "indicators": indicators_list
            })
            print("Added action:", action_name)

        stage['actions'] = stage_actions
        out.append(stage)
        stage = {}

    with open("out/output.json", "w") as f:
        f.write(json.dumps(out))