from src.DocumentFactory import DocumentFactory
from src.DocumentStore import DocumentStore
from langchain_core.documents import Document
from src.AnyrunParser import AnyrunParser
from src.QAModel import QAModel
from src.VTParser import VTParser
import os
from src.build_context import build_context
from src.TextGenerationModel import TextGenerationModel 
from src.STIXParser import STIXParser
import json
from src.SummarizationModel import SummarizationModel
from src.group_attack_patterns import group_attack_patterns

CONFIG_FILE = "config.json"
documents:list[Document] = []

documents_dir = "./documents/other"
for file in os.listdir(documents_dir):
    documents.extend(DocumentFactory.from_file(os.path.join(documents_dir, file)))

documents_dir = "./documents/anyrun"
for file in os.listdir(documents_dir):
    if file.endswith(".txt"):
        parsed_content = AnyrunParser.parse_txt(os.path.join(documents_dir, file))
        documents.extend(DocumentFactory.from_content(parsed_content))
    else:
        documents.extend(DocumentFactory.from_file(os.path.join(documents_dir, file)))

documents_dir = "./documents/vt"
for file in os.listdir(documents_dir):
    if file.endswith("json"):
        parsed_content = VTParser.parse_json(os.path.join(documents_dir, file))
        documents.extend(DocumentFactory.from_content(parsed_content))
    else:
        documents.extend(DocumentFactory.from_file(os.path.join(documents_dir, file)))

config = json.loads(open(CONFIG_FILE).read())
stix_parser = STIXParser()
stix_parser.parse(config['STIX_FILE'])
attack_patterns = stix_parser.extract_attack_patterns()
malware_patterns = stix_parser.extract_malware()
indicators_patterns = stix_parser.extract_indicators()

for malware in malware_patterns:
    documents.extend(DocumentFactory.from_content(stix_parser.stringify_object(malware)))

for indicator in indicators_patterns:
    documents.extend(DocumentFactory.from_content(stix_parser.stringify_object(indicator)))

mitre_tactics = json.loads(open("mitre-tactics.json").read())
grouped_patterns = group_attack_patterns(mitre_tactics, attack_patterns)

docstore = DocumentStore()
docstore.ingest(documents)

query = "Given the context, does the malware necessitate a specific operating system?"
#context = docstore.search_mmr(query, k=4, fetch_k=20, lambda_mult=0.5)
context = docstore.similarity_search(query, k=3, fetch_k=10)
context = build_context(context)
print(context)
exit()
text_generation_model = TextGenerationModel('microsoft/Phi-3.5-mini-instruct')
summary_model = SummarizationModel('Falconsai/text_summarization')
qa_model = QAModel(config['MODELS']['QA'])

summary = text_generation_model.invoke(query, context)
#response = qa_model.invoke(query, summary)
print(summary)
    