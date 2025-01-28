from src.DocumentParser import DocumentParser
from src.STIXParser import STIXParser
from src.extract_iocs import extract_iocs
from src.group_attack_patterns import group_attack_patterns
import os
import warnings
import json
import uuid
import ast
from tqdm import tqdm
import dotenv
import time

MODEL = "RAG_llama3.1:8b"
FILES_DIR = "documents"
STIX_DIR = "sample_stix"
STIX_FILE = "goofy-guineapig-stix.json"

if __name__ == "__main__":
    tic = time.time()
    stix_parser = STIXParser()
    agent = DocumentParser(model=MODEL)
    attack_steps = {} # this will contain the attack graph

    """
    LLM RAG INITIALIZATION
    """
    print("Starting the script\n============\n")
    
    # ingest documents
    agent.ingest(FILES_DIR)

    """
    STIX PROCESSING
    """
    stix_parser.parse(os.path.join(STIX_DIR, STIX_FILE))
    attack_patterns = stix_parser.extract_attack_patterns()
    malware_patterns = stix_parser.extract_malware()
    indicators_patterns = stix_parser.extract_indicators()

    # load MITRE tactics
    tactics = json.loads(open("mitre-tactics.json").read())
    attack = group_attack_patterns(tactics, attack_patterns)

    # Save attack dictionary to JSON
    json.dump(attack, open('attack.json', 'w'))

    """
    RAG ENRICHMENT & GRAPH GENERATION
    """
    progress = tqdm(total=len(tactics), desc="Building the attack graph")
    for state_id,tactic in enumerate(attack.keys()):
        progress.update(1)
        attack_steps[state_id] = {}
        # the graph starts with an initial state, then ech tactic is a state
        if state_id == 0:
            # initial state
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
        else:
            # next states
            query = f"""
            Considering the actual attack tactic "{tactic}", provide how the tactic is implemented by the malware.
            This state must take into account the information provided in the documents and the attack patterns already described.
            Do not provide information that is not present in the documents.
            Do not provide an introduction to the answer.
            """
        response = agent.ask(query)
        attack_steps[state_id]["state"] = response

        # ask for a title
        query = f"""
        Given the state's description {attack_steps[state_id]["state"]}, generate a brief but exhaustive title for the state.
        Do not insert any markdown or formatting.
        """
        response = agent.ask(query)
        attack_steps[state_id]["title"] = response

        # ask for the actions required to move to the next state
        attack_steps[state_id]["actions"] = []
        for ap in attack[tactic]:
            query = f"""
            Considering the attack pattern with name "{ap['name']}" and description "{ap['description']}",
            provide a brief summary of what the malware is doing. The format you must strictly follow is:"""+"""
            {
                "summary": "<brief summary>",
                "indicators": "<indicators>"
            }
            The summary should be a short but complete text. Regarding the summary, you can use the information in the description field of the attack pattern, but you must
            provide a summary that is based on the information in the documents. The summary must be precise and fit the content of the documents.
            In this summary, correlate those information with the indicators, possibly defining their meaning and origin.
            You must insert only indicators existing in the documents.
            Each indicator must be separated by a | character. If no indicators are present, write "no indicators found".
            each indicator must be a complete sentence where there is the ioc and the context in which it is found.
            """
            response = agent.ask(query)
            try:
                res = ast.literal_eval(response)
            except:
                res = {"summary": "No summary found", "indicators": "No indicators found"}

            attack_steps[state_id]["actions"].append({
                "name": ap['name'],
                "summary": res['summary'],
                "indicators": res['indicators']
            })

        with open("attack_steps.json", "w") as f:
            json.dump(attack_steps, f)

    progress.close()
    toc = time.time()
    print(f"Script completed in {toc-tic} seconds")