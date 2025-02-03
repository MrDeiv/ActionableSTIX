import time
import json
from tqdm import tqdm
from src.DocumentStore import DocumentStore
from src.STIXParser import STIXParser
from src.group_attack_patterns import group_attack_patterns
from langchain_core.documents import Document
from src.QAModel import QAModel
from src.SummarizationModel import SummarizationModel
from src.TextGenerationModel import TextGenerationModel
import os

CONFIG_FILE = "config.json"

def build_context(documents:list[Document]) -> str:
    context = ""
    for doc in documents:
        context += doc.page_content + "\n"
    return context

if __name__ == "__main__":

    # tic
    tic = time.time()
    print("Starting...")

    # final result
    output = {}
    
    # load config
    config = json.loads(open(CONFIG_FILE).read())

    """
    +----------------------------+
    |                            |
    |  STIX pre-processing       |
    |                            |
    +----------------------------+
    """

    # load original stix
    stix_parser = STIXParser()
    
    stix_parser.parse(config['STIX_FILE'])
    attack_patterns = stix_parser.extract_attack_patterns()
    malware_patterns = stix_parser.extract_malware()
    indicators_patterns = stix_parser.extract_indicators()

    # load MITRE tactics
    mitre_tactics = json.loads(open("mitre-tactics.json").read())

    # group attack patterns by tactic
    # now each tactic has a list of attack patterns associated with it
    grouped_patterns = group_attack_patterns(mitre_tactics, attack_patterns)

    """
    +----------------------------+                                                                   
    |                            |                                                                   
    |  Vectorize documents       |                                                                   
    |                            |                                                                   
    +----------------------------+
    """

    # set up document store
    docstore = DocumentStore()

    # ingest documents: chunk and embeds into vector store
    docstore.ingest(config['DOCUMENTS_DIR'])

    """
    +----------------------------+
    |                            |
    |  Chat Models               |
    |                            |
    +----------------------------+
    """

    # set models
    qa_model = QAModel(config['MODELS']['QA'])
    summary_model = SummarizationModel(config['MODELS']['SUMMARIZATION'])
    text_generation_model = TextGenerationModel(config['MODELS']['TEXT_GENERATION'])

    """
    +----------------------------+
    |                            |
    |  Build Actionable STIX     |
    |                            |
    +----------------------------+
    """

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

    progress = tqdm(total=len(grouped_patterns.keys()), desc="Defining Actionable STIX")
    for state_id, mitre_tactic in enumerate(grouped_patterns.keys()):
        progress.update(1)
        output[state_id] = {}

        if state_id == 0:
            # Malware name
            question = "Given the context, which is the name given to the malware described in the documents?"
            docs = docstore.search_mmr(question, k=config['k'], lambda_mult=config['LAMBDA'])
            context = build_context(docs)
            summary = summary_model.invoke(context)
            malware_name = qa_model.invoke(question, summary)

            # connectivity
            question = "Given the context, must the machine be connected to internet to complete the malware attack?"
            docs = docstore.search_mmr(question, k=config['k'], lambda_mult=config['LAMBDA'])
            context = build_context(docs)
            summary = summary_model.invoke(context)
            connectivity = text_generation_model.invoke(question, summary)

            # delivery
            question = "Given the context, how the malware is delivered to the target machine?"
            docs = docstore.search_mmr(question, k=config['k'], lambda_mult=config['LAMBDA'])
            context = build_context(docs)
            summary = summary_model.invoke(context)
            delivery = text_generation_model.invoke(question, summary)

            answer = {
                "malware_name": malware_name,
                "connectivity": connectivity,
                "delivery": delivery
            }
            
        else:
            continue
        #response = chat.invoke(query)
        output[state_id]['summary'] = answer

        # enrich atatck patters
        output[state_id]['actions'] = []
        progress2 = tqdm(total=len(grouped_patterns.keys()), desc="Enriching Attack Patterns")
        for ap in grouped_patterns[mitre_tactic]:
            progress2.update(1)

            # enrich attack pattern
            query = f"""
            Given the attack pattern's name: "{ap['name']}", and description: "{ap['description']}",
            provide a more detailed description of the attack pattern, adding information gathered from the
            documents.
            Do not include the original description and name in the answer.
            """
            docs = docstore.search_mmr(query, k=config['k'], lambda_mult=config['LAMBDA'])
            context = build_context(docs)
            summary = summary_model.invoke(context)
            detailed_description = text_generation_model.invoke(query, context)

            # trying to extract IoCs
            query = f"""
            Given this description: "{ap['description']}", provide a list
            of Indicators of Compromise (IoCs) that can be used to detect this attack pattern.
            You must insert only the IoCs in the answer. Those IoCs must be extracted from the documents.
            You must insert only technical indicators, such as IPs, URLs, hashes, etc.
            """
            docs = docstore.search_mmr(query, k=config['k'], lambda_mult=config['LAMBDA'])
            context = build_context(docs)
            iocs = text_generation_model.invoke(query, context)

            output[state_id]['actions'].append({
                "name": ap['name'],
                "description": detailed_description,
                "indicators": iocs
            })

        progress2.close()
        break

    progress.close()
    # save output
    output_file = os.path.join(config['OUTPUT_DIR'], config['OUTPUT_FILE'])
    json.dump(output, open(output_file, 'w'))

    # toc
    toc = time.time()
    # to minutes
    print(f"Time Elapsed: {(toc - tic)/60:.2f} min")

