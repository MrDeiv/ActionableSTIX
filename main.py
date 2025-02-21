import json
import os

from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from src.QAModel import QAModel
from src.SentimentAnalysisModel import SentimentAnalysisModel
from src.SummarizationModel import SummarizationModel
from src.TextGenerationModel import TextGenerationModel
from src.build_context import build_context
from src.group_attack_patterns import group_attack_patterns
from src.DocumentFactory import DocumentFactory
from src.stores.DocumentStore import DocumentStore
from src.STIXParser import STIXParser
from nltk.corpus import stopwords
import re
from langchain_core.documents import Document
import nltk
from src.agents.get_mitre_technique import get_mitre_technique

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
nltk.download('stopwords')

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
    qa_model = QAModel(config['MODELS']['QA'])
    text_generation_model = TextGenerationModel(
        config['MODELS']['TEXT_GENERATION'],
        max_new_tokens=256)
    #summary_model = SummarizationModel(config['MODELS']['SUMMARIZATION'])
    sentiment_model = SentimentAnalysisModel(
        config['MODELS']['TEXT_GENERATION'],
        max_new_tokens=64)
    mitre_model = TextGenerationModel(
        config['MODELS']['TEXT_GENERATION'],
        max_new_tokens=128
    )

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
    
    print("[+] Computed pre-conditions for the malware to run")
    target_hash = 'a21dec89611368313e138480b3c94835'
    anyrun_json = json.load(open(f"./documents/anyrun/{target_hash}.json", encoding='utf-8'))
    anyrun_mitre = anyrun_json['mitre']
    stop_words = set(stopwords.words('english'))
    sent_trasf = SentenceTransformer('all-MiniLM-L6-v2')
    names_embeddings = []
    for tactic in anyrun_mitre:
        name = tactic['name']
        name_normalized = name
        #name_normalized = (' '.join([word for word in name.split() if word not in stop_words])).lower()
        names_embeddings.append(sent_trasf.encode(name_normalized))
    
    anyrun_ioc = json.load(open(f"./documents/anyrun/{target_hash}_ioc.json", encoding='utf-8'))

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
            while len(techs) < config['SIMILAR_TECHNIQUES']:
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
            try:
                mitre_id, mitre_name = technique.split(" - ")
            except:
                print("Error:", technique)
                mitre_id = ""
                mitre_name = ""

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
            print("[+] Added action:", action_name)

        stage['actions'] = stage_actions
        out.append(stage)
        stage = {}

        # save the output every time a tactic is processed
        with open(f"{config['OUTPUT_DIR']}/output.json", "w") as f:
            f.write(json.dumps(out))

    