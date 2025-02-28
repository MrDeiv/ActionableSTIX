import json, os, dotenv, time, asyncio, re, logging, warnings
from tqdm import tqdm
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk import word_tokenize
from sentence_transformers import SentenceTransformer
from bs4 import BeautifulSoup
from markdown import markdown

from src.STIXParser import STIXParser
from src.group_attack_patterns import group_attack_patterns
import uuid

from src.DocumentFactory import DocumentFactory
from src.QAModel import QAModel
from src.ListParser import ListParser

# LangChain
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnableSequence, Runnable, RunnablePassthrough
from langchain.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_ollama import ChatOllama
from langchain.output_parsers import BooleanOutputParser
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq

from src.stores.DocumentStore import DocumentStore
import logging.config

def remove_markdown(text: str) -> str:
    mk = markdown(text)
    warnings.filterwarnings("ignore", category=UserWarning, module='bs4')
    return ''.join(BeautifulSoup(mk, features="html.parser").findAll(text=True))

logging.config.dictConfig({
    'version': 1,
    'disable_existing_loggers': True,
})

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

    logger = logging.getLogger(__name__)
    log_file = os.path.join(config["LOGS_DIR"], "app.log")
    logging.basicConfig(level=logging.INFO, filename=log_file, filemode="w", format="[%(asctime)s %(levelname)s] %(message)s")
    logger.info("Application started")

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
    logging.info(f"Documents loaded: {len(documents)}")

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
    docs = ensemble_retriever.invoke(retriever_query)

    logger.info(f"Connectivity computed using {len(docs)} documents. The documents are:\n{docs}")

    connectivity_summary = await chain_summary.ainvoke({"context": docs})
    connectivity_summary = remove_markdown(connectivity_summary)    

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
    docs = ensemble_retriever.invoke(retriever_query)
    logger.info(f"OS computed using {len(docs)} documents. The documents are:\n{docs}")
    os_summary = await chain_os.ainvoke({"context": docs})
    os_summary = remove_markdown(os_summary)

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
    You MUST state which software vulnerability the malware exploits.
    If the are no vulnerabilities exploited, you MUST determine if the malware disguises itself as a legitimate software.
    If the answer is not directly stated in the text, you MUST infer the answer.
    
    The context is:
    {context}
    """

    chain_vuln = RunnableSequence(
        first=ChatPromptTemplate.from_template(query),
        middle=[ollama_llm],
        last=StrOutputParser()
    )

    retriever_query = "Which software vulnerability does the malware exploit? Does the malware disguise itself as a legitimate software?"
    docs = ensemble_retriever.invoke(retriever_query)
    logger.info(f"Vulnerability computed using {len(docs)} documents. The documents are:\n{docs}")
    vuln_summary = await chain_vuln.ainvoke({"context": docs})
    vuln_summary = remove_markdown(vuln_summary)
    print("[+] Vulnerability computed")

    state['id'] = str(uuid.uuid4())
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

    logging.info(f"Inserted state with id: {state['id']}")

    ###################################################

    stop_words = set(stopwords.words('english'))

    mitre_techniques = json.loads(open("mitre/mitre-techniques.json").read())
    qa_llm = QAModel(model=config['MODELS']['QA'])

    # refinement pipeline
    query_refinement_template = """
    Given this context:
    {context}.
    
    You must state how the action is performed. The action is: 
    {action} 
    """
    refinement_llm = ChatOllama(model="llama3.1:8b", num_predict=256, temperature=0)
    chain_refinement = RunnableSequence(
        first=ChatPromptTemplate.from_template(query_refinement_template),
        middle=[refinement_llm],
        last=StrOutputParser()
    )

    # pre-conditions pipeline
    query_summary = """
    Given the following set of actions: {context}.
    Suppose all the actions are performed in the same environment and succeed.
    You MUST determine which traces are left behind by the actions. These traces must be permanent and visible.
    You MUST provide a list of traces, DO NOT provide any additional information.
    """
    summary_llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)
    chain_precond = RunnableSequence(
        first=ChatPromptTemplate.from_template(query_summary),
        middle=[summary_llm],
        last=ListParser()
    )

    for tactic in grouped_patterns:
        # each iteration is a milestone

        print("[+] Processing tactic:", tactic)
        logging.info(f"Processing step relative to tactic: {tactic}")
        interesting_techniques = mitre_techniques[tactic]['techniques']

        state['attack_steps'] = []
        for action in grouped_patterns[tactic]:
            # each iteration is an attack step
            action_name = action['name']
            logging.info(f"+ Processing action: {action_name}")
            action_description = action['description']

            sentence_transformer = SentenceTransformer(config['MODELS']['SENTENCE_TRANSFORMER'], token=os.getenv("HF_API_KEY"))
            
            # prepare embedding
            summary_text = f"{action_name}: {action_description}"
            action_nlp = " ".join([word for word in word_tokenize(summary_text) if word.lower() not in stop_words])
            action_vector = sentence_transformer.encode(action_nlp)
            
            logging.info(f"++ Embedding computed for: {action_name}. The vector has shape: {action_vector.shape}")

            # find the most similar techniques
            scores = {}
            # for each technique, compute the similarity with the action
            # then select the N highest similarity scores
            for technique in interesting_techniques:
                technique_name = technique['name']
                technique_description = technique['description']
                
                summary_tech = f"{technique_name}: {technique_description}"
                technique_nlp = " ".join([word for word in word_tokenize(summary_tech) if word.lower() not in stop_words])
                technique_vector = sentence_transformer.encode(technique_nlp)

                similarity = sentence_transformer.similarity(action_vector, technique_vector)
                scores[technique_name] = similarity
                logging.info(f"++ Similarity between [{action_name}] and [{technique_name}]: {similarity}")
            
            # sort the scores
            scores = sorted(scores.items(), key=lambda item: item[1], reverse=True)[:config['N_TECHNIQUES']]
            logger.info(f"++ Scores: {scores}")
            # get the candidate techniques
            action_mitre_technique_candidated = [score[0] for score in scores]

            assert len(action_mitre_technique_candidated) > 0, "No similar techniques found"
            #logging.info(f"++ Similar techniques found (in {attempts} attemps):\n{action_mitre_technique_candidated}")
            logging.info(f"++ Similar techniques found:\n{action_mitre_technique_candidated}")

            # given the set of most similar techniques, select the most appropriate one using the QA model
            context = "\n".join(action_mitre_technique_candidated) if action_mitre_technique_candidated else "Not provided"
            query = """
            You MUST select the most appropriate MITRE Technique for the action called: \n"""+action_name+"""\n
            and description: \n"""+action_description+"""\n
            You MUST fit the action with the most appropriate MITRE Technique, DO NOT add any additional information.
            You MUST select one choice, DO NOT infer the answer.
            Each choice is separated by a new line, DO NOT truncate the choices.
            """.format(context=context)
            
            logging.info(f"++ Querying the QA model for action {action_name} with the following context:\n{context}")
            action_technique_name = qa_llm.invoke(query, context).strip()
            logging.info(f"++ QA selected technique: {action_technique_name}")

            try:
                action_technique_id = list(filter(lambda x: x['name'] == action_technique_name, interesting_techniques))[0]['id']
                action_technique_description = list(filter(lambda x: x['name'] == action_technique_name, interesting_techniques))[0]['description']
            except IndexError:
                # fallback
                logging.warning(f"++ Technique not found: {action_technique_name} in {action_mitre_technique_candidated}. Trying to refine the answer.")
                if '\n' in action_technique_name:
                    action_technique_name = action_technique_name.split('\n')[0]
                    action_technique_id = list(filter(lambda x: x['name'] in action_technique_name, interesting_techniques))[0]['id']
                    action_technique_description = list(filter(lambda x: x['name'] in action_technique_name, interesting_techniques))[0]['description']
                    logging.info(f"++ Technique found after refining: {action_technique_name}")
                else:
                    # worst case
                    action_technique_id = "Unknown"
                    action_technique_description = "Unknown"
                    print("[!] Technique not found:", action_technique_name, "in:", action_mitre_technique_candidated)
                    logging.error(f"++ Technique not found: {action_technique_name} in {action_mitre_technique_candidated}")
                
            # MITRE reference
            technique = {
                "id": action_technique_id,
                "name": action_technique_name.capitalize(),
                "description": action_technique_description
            }
            
            # refine the action description using the MITRE technique as reference
            query_refinement = """
            Given this MITRE technique: {context}.\nYou MUST state how the action: {action}, fit the given technique.
            DO NOT insert any introduction or additional information.
            DO NOT cite the documents.
            DO NOT add any markdown.
            DO NOT insert any code.
            You MUST provide only a detailed description.
            """.format(context=action_technique_name, action=action_name)

            docs = ensemble_retriever.invoke(query_refinement)
            logging.info(f"++ Refining the action: {action_name} using {len(docs)} documents:\n{docs}")
            refined_description = chain_refinement.invoke({
                "context": "\n".join([doc.page_content for doc in docs]),
                "action": action_name + " " + action_description
            })

            # pre-conditions
            action['pre-conditions'] = []
            query_preconditions = """
            Given the following context: {context}.

            You MUST determine the pre-conditions for the action: {action}.
            You MUST provide a list of pre-conditions, DO NOT provide any additional information.
            """.format(context=refined_description, action=action_name)

            pre_conditions = chain_precond.invoke({"context": query_preconditions})
            logging.info(f"++ Pre-conditions computed for action: {action_name}")

            # post-conditions
            action['post-conditions'] = []
            query_postconditions = """
            Given the following context: 
            {context}

            Suppose all the actions are performed in the same environment and succeed.
            You MUST determine which traces are left behind by the actions. These traces must be permanent and visible.
            You MUST provide a list of traces, DO NOT provide any additional information.
            """.format(context=refined_description)

            post_conditions = chain_precond.invoke({"context": query_postconditions})
            logging.info(f"++ Post-conditions computed for action: {action_name}")

            # action
            refined_description = remove_markdown(refined_description)
            actions = {
                "id": str(uuid.uuid4()),
                "name": action_name,
                "description": refined_description,
                "mitre_technique": technique,
                "pre-conditions": pre_conditions,
                "post-conditions": post_conditions,
                "indicators": []
            }

            # add actions to the attack step
            state['attack_steps'].append(actions)

        output.append(state)

        # next attack step
        """ previous_actions = ("#"*10).join([f"{action['name']} - {action['description']}" for action in state['actions']])
        
        logging.info(f"+ Computing pre-conditions for the next attack step using the following actions:\n{previous_actions}")
        
        pre_conditions = chain_precond.invoke({"context": previous_actions})
        print("[+] Post-conditions computed") """

        state = {}
        state['id'] = str(uuid.uuid4())
        """ state['pre-conditions'] = []
        state['pre-conditions'].extend(pre_conditions)
        state['actions'] = [] """

    # save output
    logging.info("Saving output")
    with open(os.path.join(config["OUTPUT_DIR"], config["OUTPUT_FILE"]), "w") as f:
        json.dump(output, f)

    logger.info("Application finished")


if __name__ == "__main__":
    asyncio.run(main())