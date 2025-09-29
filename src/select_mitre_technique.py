from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain_core.runnables import RunnableSequence
from langchain.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser

def select_mitre_technique():
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
    qa_llm = ChatOllama(model="gemma2:9b", temperature=0)
    chain_qa = RunnableSequence(
        first=ChatPromptTemplate.from_template(qa_template),
        middle=[qa_llm],
        last=StrOutputParser()
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
            action_name = action_name.replace(malware_name, "the malware") # generalize the action name
            
            logging.info(f"+ Processing action: {action_name}")
            action_description = action['description']
            action_description = action_description.replace(malware_name, "the malware") # generalize the action description

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

            action_technique_name = chain_qa.invoke({"context": context, "action": action_name, "description": action_description}).strip()
            action_technique_name = remove_markdown(action_technique_name)

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

            try:
                action_technique_id = list(filter(lambda x: x['name'] == action_technique_name, interesting_techniques))[0]['id']
                action_technique_description = list(filter(lambda x: x['name'] == action_technique_name, interesting_techniques))[0]['description']
            except:
                # fallback to the first technique if the selected technique is not in the list
                action_technique_name = action_mitre_technique_candidated[0]
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
            logging.info(f"++ Pre-conditions computed for action: {action_name} using {len(docs)} documents:\n{docs}")

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
            You MUST determine which are the consequences of the action. These consequences must be permanent and visible.
            You MUST provide a list of consequences, DO NOT provide any additional information.
            """.format(context=docs)

            post_conditions = chain_postcond.invoke({"context": query_postconditions})
            logging.info(f"++ Post-conditions computed for action: {action_name} using {len(docs)} documents:\n{docs}")

            # indicators
            indicators = chain_indicators.invoke({"context": "\n".join(iocs), "action": action_name})
            logging.info(f"++ Indicators computed for action: {action_name}. The indicators are:\n{indicators}")

            # refine pre-conditions
            pre_conditions = [remove_markdown(pre) for pre in pre_conditions]
            
            for pre in pre_conditions:
                # fix generation errors
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
            refined_description = chain_rephrase.invoke({"context": refined_description})

            actions = {
                "id": str(uuid.uuid4()),
                "name": action_name,
                "description": refined_description,
                "mitre_technique": technique,
                "pre-conditions": pre_conditions,
                "post-conditions": post_conditions,
                "indicators": indicators
            }

            # add actions to the attack step
            state['attack_steps'].append(actions)

        output.append(state)

        state = {}
        state['id'] = str(uuid.uuid4())

    logging.info("Created output with %d milestones", len(output))
    console.print(f"Created output with {len(output)} milestones", style="bold green")