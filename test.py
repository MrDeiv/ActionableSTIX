import json
from src.STIXParser import STIXParser
from src.group_attack_patterns import group_attack_patterns
from src.DocumentStore import DocumentStore
from langchain_core.documents import Document
from src.QAModel import QAModel
from src.SummarizationModel import SummarizationModel
from src.TextGenerationModel import TextGenerationModel
from tqdm import tqdm

CONFIG_FILE = "config.json"

def build_context(documents:list[Document]) -> str:
    context = ""
    for doc in documents:
        context += doc.page_content + "\n"
    return context

config = json.loads(open(CONFIG_FILE).read())
stix_parser = STIXParser()
stix_parser.parse(config['STIX_FILE'])
attack_patterns = stix_parser.extract_attack_patterns()
malware_patterns = stix_parser.extract_malware()
indicators_patterns = stix_parser.extract_indicators()

mitre_tactics = json.loads(open("mitre-tactics.json").read())
grouped_patterns = group_attack_patterns(mitre_tactics, attack_patterns)

docstore = DocumentStore()

# ingest documents: chunk and embeds into vector store
docstore.ingest(config['DOCUMENTS_DIR'])

qa_model = QAModel(config['MODELS']['QA'])
summary_model = SummarizationModel(config['MODELS']['SUMMARIZATION'])
text_generation_model = TextGenerationModel(config['MODELS']['TEXT_GENERATION'])

progress = tqdm(total=len(malware_patterns), desc="Ingesting Malwares")
for malware in malware_patterns:
    progress.update(1)
    docstore.add_text(stix_parser.stringify_object(malware))
progress.close()

progress = tqdm(total=len(indicators_patterns), desc="Ingesting Indicators")
for indicator in indicators_patterns:
    progress.update(1)
    docstore.add_text(stix_parser.stringify_object(indicator))
progress.close()

tactic = grouped_patterns['persistence']
for attack_pattern in tactic:
    ap_name = attack_pattern['name']
    ap_description = attack_pattern['description']

    query = f"""
    Given the attack pattern's name: "{ap_name}", and description: "{ap_description}",
    provide a more detailed description of the attack pattern, adding information gathered from the
    documents.
    Do not include the original description and name in the answer.
    """
    docs = docstore.search_mmr(query, k=config['k'], lambda_mult=config['LAMBDA'])
    context = build_context(docs)
    summary = summary_model.invoke(context)
    answer = text_generation_model.invoke(query, context)

    query = f"""
    Given this description: "{ap_description}", provide a list
    of Indicators of Compromise (IoCs) that can be used to detect this attack pattern.
    You must insert only the IoCs in the answer. Those IoCs must be extracted from the documents.
    You must insert only technical indicators, such as IPs, URLs, hashes, etc.
    """
    docs = docstore.search_mmr(query, k=config['k'], lambda_mult=config['LAMBDA'])
    context = build_context(docs)
    text_generation_model2 = TextGenerationModel(config['MODELS']['TEXT_GENERATION'], max_new_tokens=512)
    answer = text_generation_model2.invoke(query, context)
    print(answer)
    exit()