import os
from langchain_core.runnables import RunnableSequence
from langchain.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from sentence_transformers import SentenceTransformer
import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords
from bs4 import BeautifulSoup
from markdown import markdown
from tqdm import tqdm
import json
from mitre_distance import compute_graph_distance
import warnings
import networkx as nx
import time
import logging
from huggingface_hub import snapshot_download

snapshot_download(repo_id="sentence-transformers/all-MiniLM-L6-v2", repo_type="model")

warnings.filterwarnings("ignore", category=DeprecationWarning) 

SAMPLES_FOLDER = 'ablation/mitre/samples'
GT_FOLDER = 'ablation/mitre/'
MITRE_SOURCE = 'mitre/mitre-techniques.json'
MITRE_GRAPH_FILE = 'mitre/mitre_graph.json'
CONFIG_FILE = "config/config.json"
LOG_FILE = "ablation/mitre/abl_mitre.log"

MODELS = [ 
    #'gemma2:9b', 
    #'gpt-oss:20b',
    #'deepseek-r1:1.5b', 
    #'deepseek-r1:7b', 
    #'deepseek-r1:8b', 
    #'gemma3:270m', 
    #'gemma3:1b', 
    #'gemma3:4b', 
    #'gemma3:12b', 
    #'qwen3:0.6b', 
    #'qwen3:1.7b',
    #'qwen3:4b',
    'qwen3:8b',
    'qwen3:14b',
    'llama3.1:8b']

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt_tab')

#logging.basicConfig(filename=LOG_FILE, level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')

def remove_markdown(text: str) -> str:
    mk = markdown(text)
    return ''.join(BeautifulSoup(mk, features="html.parser").find_all(text=True))

if __name__ == "__main__":
    config = json.load(open(CONFIG_FILE))
    stop_words = set(stopwords.words('english'))
    selected_interaction_level = config['SELECTED_INTERACTION_LEVEL']
    G = nx.node_link_graph(json.load(open(MITRE_GRAPH_FILE, 'r')))

    mitre = json.load(open(MITRE_SOURCE, encoding='utf-8'))
    gt = json.load(open(os.path.join(GT_FOLDER, 'gt.json'), encoding='utf-8'))

    # ablation
    # qa pipeline
    qa_template = """
    Given this list of MITRE Techniques:
    {context}.

    You MUST select the most appropriate MITRE Technique for the action called:
    {action}.
    With description:
    {description}.
    You MUST select one choice from the list, DO NOT add any additional information.
    Each choice is separated by a new line, DO NOT truncate the choices.
    You MUST select one choice, DO NOT infer the answer.
    """

    results = []
    for model in MODELS:
        logging.error(f"Processing model: {model}")
        print(f"[-] Processing model: {model}")
        partial = {
            'model': model,
        }
        qa_llm = ChatOllama(model=model, temperature=0)
        chain_qa = RunnableSequence(
            first=ChatPromptTemplate.from_template(qa_template),
            middle=[qa_llm],
            last=StrOutputParser()
        )

        correct = 0
        human_in_the_loop_calls = 0
        total = 0
        partial['total'] = 0
        partial['average_match_time'] = 0
        partial['average_distance'] = 0
        partial['correct'] = 0
        partial['human_in_the_loop_calls'] = 0
        progress = tqdm(gt, desc=f"Model {model} Progress", unit="case")
        try:
            for case in gt:
                start = time.time()
                action_name = case['name']
                action_description = case['description']
                interesting_techniques = mitre[case['mitre_tactic']]['techniques']
                distance = 0

                sentence_transformer = SentenceTransformer(config['MODELS']['SENTENCE_TRANSFORMER'], local_files_only=True)
                # prepare embedding
                summary_text = f"{action_name}: {action_description}"
                action_nlp = " ".join([word for word in word_tokenize(summary_text) if word.lower() not in stop_words])
                action_vector = sentence_transformer.encode(action_nlp)

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

                scores = sorted(scores.items(), key=lambda item: item[1], reverse=True)[:config['N_TECHNIQUES']]
                action_mitre_technique_candidated = [score[0] for score in scores]

                if len(action_mitre_technique_candidated) > 0:
                    context = "\n".join(action_mitre_technique_candidated) if action_mitre_technique_candidated else "Not provided"
                    query = """
                    You MUST select the most appropriate MITRE Technique for the action called: \n"""+action_name+"""\n
                    and description: \n"""+action_description+"""\n
                    You MUST fit the action with the most appropriate MITRE Technique, DO NOT add any additional information.
                    You MUST select one choice, DO NOT infer the answer.
                    Each choice is separated by a new line, DO NOT truncate the choices.
                    """.format(context=context)

                    action_technique_name = chain_qa.invoke({"context": context, "action": action_name, "description": action_description}).strip()
                    action_technique_name = remove_markdown(action_technique_name)

                    if "\n" in action_technique_name:
                        # fallback to the first line since the QA model returns multiple lines
                        action_technique_name = action_technique_name.split("\n")[0]

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
                                    break
                        if human_in_the_loop:
                            break

                    if human_in_the_loop:
                        human_in_the_loop_calls += 1

                    try:
                        action_technique_id = list(filter(lambda x: x['name'] == action_technique_name, interesting_techniques))[0]['id']
                        action_technique_description = list(filter(lambda x: x['name'] == action_technique_name, interesting_techniques))[0]['description']
                    except:
                        # fallback to the first technique if the selected technique is not in the list
                        action_technique_name = action_mitre_technique_candidated[0]
                        action_technique_id = list(filter(lambda x: x['name'] == action_technique_name, interesting_techniques))[0]['id']
                        action_technique_description = list(filter(lambda x: x['name'] == action_technique_name, interesting_techniques))[0]['description']

                    end = time.time()
                    elapsed_time = end - start
                    partial['average_match_time'] = partial['average_match_time'] + elapsed_time
                    #logging.error(f"Elasped time for model {model}: {elapsed_time} seconds")

                    is_correct = action_technique_name.lower() == case['name'].lower()
                    
                    if is_correct and not human_in_the_loop:
                        correct += 1
                    elif not is_correct and not human_in_the_loop:
                        # compute the graph distance between the selected technique and the ground truth
                        distance, path = compute_graph_distance(G, action_technique_id, case['mitre_technique'])
                        #logging.error(f"Computed graph distance between selected technique '{action_technique_name}' and ground truth '{case['name']}': {distance}")
                        partial['average_distance'] = partial['average_distance'] + distance

                total += 1
                progress.update(1)
            progress.close()
            partial['total'] = total
            partial['correct'] = correct
            partial['human_in_the_loop_calls'] = human_in_the_loop_calls
        except Exception as e:
            #logging.error(f"Error processing model {model}: {e}")
            progress.close()

        errors = partial['total'] - partial['correct'] - partial['human_in_the_loop_calls']
        partial['average_match_time'] = partial['average_match_time'] / partial['total'] 
        partial['average_distance'] = partial['average_distance'] / errors if errors > 0 else 0
        partial['accuracy'] = partial['correct'] / partial['total'] 
        partial['human_in_the_loop_ratio'] = partial['human_in_the_loop_calls'] / partial['total']
        results.append(partial)

        with open(os.path.join(GT_FOLDER, f"partial_{model.replace(':', '_')}.json"), 'w', encoding='utf-8') as f:
            json.dump(partial, f, indent=4)

    with open(os.path.join(GT_FOLDER, f"ablation_results.json"), 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)