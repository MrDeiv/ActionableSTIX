import json, os, dotenv, time, asyncio, re, logging, warnings
from tqdm import tqdm
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk import word_tokenize
from sentence_transformers import SentenceTransformer
from bs4 import BeautifulSoup
from markdown import markdown
from rich import print as rprint
from rich.console import Console

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
    return ''.join(BeautifulSoup(mk, features="html.parser").find_all(text=True))

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
    console = Console()

    # load config
    config = json.load(open(CONFIG_FILE))

    interaction_levels = config['INTERACTION_LEVELS'].keys()
    selected_interaction_level = config['SELECTED_INTERACTION_LEVEL']

    assert selected_interaction_level in interaction_levels, f"Invalid interaction level: {selected_interaction_level}. Valid levels are: {interaction_levels}"
    
    interaction_score = config['INTERACTION_LEVELS'][selected_interaction_level]

    output = [] # output variable

    logger = logging.getLogger(__name__)
    log_file = os.path.join(config["OUTPUT_DIR"], "app.log")
    logging.basicConfig(level=logging.INFO, filename=log_file, filemode="w", format="[%(asctime)s %(levelname)s] %(message)s")
    logger.info("Application started with interaction level: %s (%f)", selected_interaction_level, interaction_score)
    console.print(f"Application started with interaction level: {selected_interaction_level} (thr.: {interaction_score})", style="bold green")

    # load files into documents
    directory = os.path.join(config["DOCUMENTS_DIR"], "other")
    documents = []
    n_files = len(os.listdir(directory))

    progress = tqdm(total=n_files, desc="Loading documents")
    for file in os.listdir(directory):
        documents.extend(DocumentFactory.from_file(os.path.join(directory, file)))
        progress.update(1)
    progress.close()

    console.print(f"Documents loaded: {len(documents)}", style="bold green")
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

    """
    STIX Parsing
    """
    # load the STIX file
    stix_parser = STIXParser()
    stix_parser.parse(config['STIX_FILE'])

    # extract the attack patterns, malware, and indicators
    attack_patterns = stix_parser.extract_attack_patterns()
    malware_patterns = stix_parser.extract_malware()
    malware_name = malware_patterns[0]['name']
    indicators_patterns = stix_parser.extract_indicators()

    iocs = [x['name'] + ": " + " ".join(x['pattern']) for x in indicators_patterns if "rule" not in x['pattern']]
    logging.info(f"Extracted {len(iocs)} IOCs from the STIX file")

    attack_patterns_used = stix_parser.get_attack_pattern_used()

    # group the attack patterns
    mitre_tactics = json.loads(open("mitre-tactics.json").read())
    grouped_patterns = group_attack_patterns(mitre_tactics, attack_patterns_used)

    # this are all the hashes mentioned in the malware's iocs, i.e., related files
    # hashes_from_indicators = get_hashes(indicators_patterns) 
    #print(hashes_from_indicators)

    state = {}
    state['id'] = str(uuid.uuid4())

    logging.info(f"Inserted state with id: {state['id']}")

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
    #refinement_llm = ChatOllama(model="llama3.1:8b", num_predict=256, temperature=0)
    refinement_llm = ChatGroq(model="gemma2-9b-it", temperature=0)
    chain_refinement = RunnableSequence(
        first=ChatPromptTemplate.from_template(query_refinement_template),
        middle=[refinement_llm],
        last=StrOutputParser()
    )

    # pre-conditions pipeline
    query_summary = """
    Given the following context: 
    
    {context}

    Suppose all the actions are performed in the same environment.
    You MUST determine which are the requirements to perform the action: {action}.
    You MUST provide a list of requirements, DO NOT provide any additional information.
    The requirements must include the environment, tools, connectivity and resources needed.
    If the requirements are not directly stated, you MUST infer the answer. If no requirements are needed, you MUST state that.
    """
    summary_llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)
    chain_precond = RunnableSequence(
        first=ChatPromptTemplate.from_template(query_summary),
        middle=[summary_llm],
        last=ListParser()
    )

    # post-conditions pipeline
    query_summary = """
    Given the following set of actions: 
    
    {context}

    Suppose all the actions are performed in the same environment and succeed.
    You MUST determine which traces are left behind by the actions. These traces must be permanent and visible.
    You MUST provide a list of traces, DO NOT provide any additional information.
    The items in the list MUST detail technical traces, such as logs, files, or network connections.
    """
    summary_llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)
    chain_postcond = RunnableSequence(
        first=ChatPromptTemplate.from_template(query_summary),
        middle=[summary_llm],
        last=ListParser()
    )

    # indicators pipeline
    query = """
    Given this list of indicators of compromise:

    {context}

    You MUST select the indicators that are related to the action: {action}.
    You MUST provide a numbered list of indicators, DO NOT provide any additional information.
    If there are no indicators related to the action, you MUST return an empty list.
    You MUST formulate the indicators in a passive sentence.
    """
    indicators_llm = ChatGroq(model="gemma2-9b-it", temperature=0)
    chain_indicators = RunnableSequence(
        first=ChatPromptTemplate.from_template(query),
        middle=[indicators_llm],
        last=ListParser()
    )

    for tactic in grouped_patterns:
        # each iteration is a milestone

        print("[+] Processing tactic:", tactic)
        logging.info(f"Processing step relative to tactic: {tactic}")
        interesting_techniques = mitre_techniques[tactic]['techniques']

        state['attack_steps'] = []
        state['pre-conditions'] = []
        state['post-conditions'] = []
        for action in grouped_patterns[tactic]:
            # each iteration is an attack step
            action_name = action['name']
            action_name = action_name.replace(malware_name, "the malware")
            
            logging.info(f"+ Processing action: {action_name}")
            action_description = action['description']
            action_description = action_description.replace(malware_name, "the malware")

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

            if "\n" in action_technique_name:
                # fallback to the first line since the QA model returns multiple lines
                action_technique_name = action_technique_name.split("\n")[0]

            logging.info(f"++ QA suggested technique: {action_technique_name}")

            # evaluate human-in-the-loop requirement
            human_in_the_loop = False
            for technique_1 in action_mitre_technique_candidated:
                for technique_2 in action_mitre_technique_candidated:
                    if technique_1 != technique_2:
                        # get the score for first and second technique
                        score_1 = [score[1] for score in scores if score[0] == technique_1][0]
                        score_2 = [score[1] for score in scores if score[0] == technique_2][0]
                        score_diff = abs(score_1 - score_2)
                        if score_diff < config['INTERACTION_LEVELS'][selected_interaction_level]:
                            human_in_the_loop = True
                            logging.info(f"++ Human-in-the-loop required for action: {action_name} due to score difference: {score_diff}")
                            break
                if human_in_the_loop:
                    break

            if human_in_the_loop:
                console.print(f"\n[[bold red]!!![/bold red]] Human decision required for action: [yellow]{action_name}[/yellow].\nPlease, select the most appropriate MITRE Technique:")
                for i, technique in enumerate(action_mitre_technique_candidated):
                    if technique == action_technique_name:
                        console.print(f"[yellow]{i+1}.[/yellow] {technique.capitalize()} ([italic yellow]suggested[/italic yellow])")
                    else:
                        console.print(f"[yellow]{i+1}.[/yellow] {technique.capitalize()}")
                print("")
                selected = int(input("> Your choice: ")) - 1
                action_technique_name = action_mitre_technique_candidated[selected]
                logging.info(f"++ Human selected technique: {action_technique_name} (index: {selected})")

            logging.info(f"++ Selected technique: {action_technique_name}")

            action_technique_id = list(filter(lambda x: x['name'] == action_technique_name, interesting_techniques))[0]['id']
            action_technique_description = list(filter(lambda x: x['name'] == action_technique_name, interesting_techniques))[0]['description']
                
            # MITRE reference
            technique = {
                "id": action_technique_id,
                "name": action_technique_name.capitalize(),
                "description": action_technique_description
            }
            
            # refine the action description using the MITRE technique as reference
            context = action_technique_name + ": " + action_technique_description
            query_refinement = """
            Given this MITRE technique: 

            {context}.

            You MUST state how the action: {action}, fit the given technique.
            DO NOT insert any introduction or additional information.
            DO NOT cite the documents.
            DO NOT add any markdown.
            DO NOT insert any code.
            You MUST provide only a detailed description.
            """.format(context=context, action=action_name)

            docs = ensemble_retriever.invoke(query_refinement)
            logging.info(f"++ Refining the action: {action_name} using {len(docs)} documents:\n{docs}")
            refined_description = chain_refinement.invoke({
                "context": "\n".join([doc.page_content for doc in docs]),
                "action": action_name + " " + action_description
            })

            # pre-conditions
            query_preconditions_retriever = """
            Given the following context:\n
            {context}.
            You MUST list what are the requirements to perform the action: {action}.
            DO NOT provide any additional information.
            The requirements must include the environment, tools, and resources needed.
            """.format(context=refined_description, action=action_name)

            docs = ensemble_retriever.invoke(query_preconditions_retriever)
            query_preconditions = """
            Given the following context: 
            {context}

            You MUST determine the pre-conditions for the action: {action}.
            You MUST provide a list of pre-conditions, DO NOT provide any additional information.
            Every item in the list MUST be a passive sentence.
            You can infer information from the context: for instance, if the action requires a specific tool, you can infer that the tool is available.
            """.format(context=docs, action=action_name)

            pre_conditions = chain_precond.invoke({"context": query_preconditions, "action": action_name})
            logging.info(f"++ Pre-conditions computed for action: {action_name}")

            # post-conditions
            query_postconditions_retriever = """
            Given the following context:\n
            {context}.
            You MUST list what are the consequences of the action: {action}.
            DO NOT provide any additional information.
            The consequences MUST be visible and technical.
            """.format(context=refined_description, action=action_name)

            docs = ensemble_retriever.invoke(query_postconditions_retriever)
            action['post-conditions'] = []
            query_postconditions = """
            Given the following context: 
            {context}

            Suppose all the actions are performed in the same environment and succeed.
            You MUST determine which the consequences of the action. These consequences must be permanent and visible.
            You MUST provide a list of consequences, DO NOT provide any additional information.
            """.format(context=docs)

            post_conditions = chain_postcond.invoke({"context": query_postconditions})
            logging.info(f"++ Post-conditions computed for action: {action_name}")

            # indicators
            indicators = chain_indicators.invoke({"context": "\n".join(iocs), "action": action_name})
            logging.info(f"++ Indicators computed for action: {action_name}. The indicators are:\n{indicators}")

            # refine pre-conditions
            pre_conditions = [remove_markdown(pre) for pre in pre_conditions]
            
            for pre in pre_conditions:
                # remove LLM typos
                if ":" in pre and len(pre.split(":")) == 1:
                    pre_conditions.remove(pre)

            # remove similar pre-conditions
            for pre_1 in pre_conditions:
                for pre_2 in pre_conditions:
                    if pre_1 != pre_2:
                        emb_1 = sentence_transformer.encode(pre_1)
                        emb_2 = sentence_transformer.encode(pre_2)
                        similarity = sentence_transformer.similarity(emb_1, emb_2)
                        if similarity > config['DUPLICATE_THRESHOLD']:
                            pre_conditions.remove(pre_2)
                            logger.info(f"++ Removed pre-condition: {pre_2}, due to similarity with: {pre_1} ({similarity})")

            pre_conditions = list(set(pre_conditions)) # remove duplicates

            # refine post-conditions
            post_conditions = [remove_markdown(post) for post in post_conditions]

            for post_1 in post_conditions:
                for post_2 in post_conditions:
                    if post_1 != post_2:
                        emb_1 = sentence_transformer.encode(post_1)
                        emb_2 = sentence_transformer.encode(post_2)
                        similarity = sentence_transformer.similarity(emb_1, emb_2)
                        if similarity > config['DUPLICATE_THRESHOLD']:
                            post_conditions.remove(post_2)
                            logger.info(f"++ Removed post-condition: {post_2}, due to similarity with: {post_1} ({similarity})")
            
            post_conditions = list(set(post_conditions)) # remove duplicates

            # action
            refined_description = remove_markdown(refined_description)
            actions = {
                "id": str(uuid.uuid4()),
                "name": action_name,
                "description": refined_description,
                "mitre_technique": technique,
                "pre-conditions": pre_conditions,
                "post-conditions": post_conditions,
                "indicators": indicators
            }

            state['pre-conditions'].extend(pre_conditions)
            state['post-conditions'].extend(post_conditions)

            # add actions to the attack step
            state['attack_steps'].append(actions)

        output.append(state)

        state = {}
        state['id'] = str(uuid.uuid4())

    # save output
    logging.info("Saving output")
    console.print(f"Saving output to [red]{os.path.join(config["OUTPUT_DIR"], config["OUTPUT_FILE"])}[/red]")    
    with open(os.path.join(config["OUTPUT_DIR"], config["OUTPUT_FILE"]), "w") as f:
        json.dump(output, f)

    logger.info("Application finished")
    console.print("Application finished!", style="bold green")


if __name__ == "__main__":
    asyncio.run(main())