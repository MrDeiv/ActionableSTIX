import json
import os
from tqdm import tqdm
from src.QAModel import QAModel
from src.SentimentAnalysisModel import SentimentAnalysisModel
from src.SummarizationModel import SummarizationModel
from src.TextGenerationModel import TextGenerationModel
from src.build_context import build_context
from src.group_attack_patterns import group_attack_patterns
from src.STIXParser import STIXParser
from langchain_core.documents import Document
from src.DocumentFactory import DocumentFactory
from src.DocumentStore import DocumentStore

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

    documents:list[Document] = []
    documents_dir = "./documents/other" # other documents (not from anyrun or vt)

    progress = tqdm(total=len(os.listdir(documents_dir)), desc="Ingesting Documents")
    for file in os.listdir(documents_dir):
        documents.extend(DocumentFactory.from_file(os.path.join(documents_dir, file)))
        progress.update(1)
    progress.close()

    docstore = DocumentStore()
    docstore.ingest(documents)

    qa_model = QAModel(config['MODELS']['QA'])
    text_generation_model = TextGenerationModel(config['MODELS']['TEXT_GENERATION'])
    summary_model = SummarizationModel(config['MODELS']['SUMMARIZATION'])
    sentiment_model = SentimentAnalysisModel(config['MODELS']['TEXT_GENERATION'])

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

    answer, score = text_generation_model.invoke(question, prompt)
    query =f"""
    Given the statement:
    {answer}

    Map it to True or False. Do not add any additional information.
    """
    connectivity_required, score = sentiment_model.invoke(answer)

    print(connectivity_required)