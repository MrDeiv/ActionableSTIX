import time
import json

from tqdm import tqdm

from src.DocumentStore import DocumentStore
from src.ChatModel import ChatModel
from src.STIXParser import STIXParser
from src.group_attack_patterns import group_attack_patterns

CONFIG_FILE = "config.json"
UPDATE_EMBEDDINGS = False

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
    ds = DocumentStore(
        model=config["MODEL_NAME"],
        collection_name=config["CHROMA_COLLECTION"]['NAME'],
        persist_directory=config["CHROMA_COLLECTION"]['DIR']
    )

    # ingest documents: chunk and embeds into vector store
    ds.ingest_directory(config['DOCUMENTS_DIR'])

    """
    +----------------------------+
    |                            |
    |  Chat Model                |
    |                            |
    +----------------------------+
    """

    # set up chat model
    chat = ChatModel(
        model_name=config["MODEL_NAME"],
        retriever=ds.retriever # retriever from document store
    )

    """
    +----------------------------+
    |                            |
    |  Build Actionable STIX     |
    |                            |
    +----------------------------+
    """

    progress = tqdm(total=len(mitre_tactics), desc="Defining Actionable STIX")
    for state_id, mitre_tactic in enumerate(mitre_tactics):
        progress.update(1)
        output[state_id] = {}

        if state_id == 0:
            query = """
                Provide a summary about the characteristics required by a system to be vulnerable to the malware described.
                Avoid adding information that is not present in the documents.
                The summary must contain:
                - the operating system(s) that are vulnerable to the malware
                - the software that is vulnerable to the malware
                - if connectivity is required
                - how the malware is delivered
                - the actions required to run the malware. For instance, if the malware is delivered via email, the user must open the attachment.
                No introduction to the answer is required.
                """+f"""
                The malware described is "{malware_patterns}".
                """
            response = chat.invoke(query)
            output[state_id]['summary'] = response
        
    # save output
    json.dump(output, open('out/output.json', 'w'))

    # toc
    toc = time.time()
    print(f"Time Elapsed: {toc - tic:.2f} s")

