# Merging Narrative and Structured Information Through LLMs for Actionable Rich Threat Representation

## Abstract
Modern cybersecurity is characterized by the abundance of data. Once a cyber threat is discovered, structured and narrative data are rapidly produced by different sources to enhance awareness and defense readiness. While Cyber Threat Intelligence (CTI) processes leverage established frameworks and formats to analyze and disseminate this data, the actionability of the results is often critical. Being able to provide insights about the evolution of a cyber attack would reduce the delay between the discovery and the implementation of the defenses. In this proposal, we present a pipeline that produces an actionable representation of a cyber attack starting from data gathered by CTI sources. We build on the concept of Attack Graph to highlight the Attack Steps and the Milestones of the attack, i.e., the pivot points of the attack, integrating this representation with the necessary Pre-conditions, the corresponding MITRE Technique, and the consequences left on the asset. This information is inferred by a Large Language Model (LLM) acting on the data of the cyber threat, which is modeled according to a Retrieval-Augmented Generation (RAG) architecture. 
We evaluated the components of our pipeline individually. For the pre-processing stages, we determined the best chunking strategy and its hyperparameters by evaluating the context precision and recall across 442 documents. Then, we selected the LLM by evaluating its ability to infer the proper MITRE Technique given a context. We tested the LLMs across 9196 technique matches. To address inaccuracies, our pipeline implements a human-in-the-loop approach.
Finally, the complete pipeline has been tested with ten case studies describing real-world threats. These cover different types of threats, from backdoors to staged downloaders delivered via phishing. We manually assessed the correctness of the generated Attack Graph, thereby demonstrating the effectiveness of the proposed pipeline in producing a novel and comprehensive structured view of attacks, which proactively improves attack prevention and defense.

## Pipeline Overview
![Pipeline Overview](docs/pipeline_schema.jpg)

In this section, we match the components described in the schema above with the code's components provided in this repository.

The heterogeneous documents and the intermediate representation (e.g., STIX file), must be inserted in the proper directories.

Users should verify the configuration file in `config/config.json` before executing the pipeline.

### Configuration File
This file allows users to customize their implementation, configuring the parameters as follows:
- `STIX_FILE`: the path of the STIX file.
- `DOCUMENTS_DIR`: the directory containing the additional files.
- `OUTPUT_DIR`: the output directory. It will contains the log file and the output JSON.
- `OUTPUT_FILE`: the name of the output file. Note that the `SELECTED_INTERACTION_LEVEL` will be inserted at the start of the filename.
- `MODELS`: the models to use in the pipeline. In particular, it requires the name of the `TEXT_GENERATION`, the `SENTENCE_TRANSFORMER`, and `MITRE_MATCH` models.
- `CHUNK_SIZE`: the size of the chunks to generate when processing the additional documents.
- `CHUNK_OVERLAP`: the overlap (%) between the chunks when processing the additional documents.
- `k`: the chunks to be retrieved by the semantic search
- `BM25_k`: the number of chunks to be retrieved by keyword search.
- `N_TECHNIQUES`: number of techniques to select to be provided to the LLM.
- `INTERACTION_LEVELS`: thresholds for the defined human interaction levels.
- `SELECTED_INTERACTION_LEVEL`: human interaction level to be used.
- `DUPLICATE_THRESHOLD`: threshold used to refine the Pre and Post conditions.

### Application File
The whole pipeline is implemented by the script `app.py`, importing relevant utilities from the `/src` folder and its subfolders.
This implementation uses STIX as intermediate representaion. To parse the STIX file and extract the interesting objects we implemented the class `src/STIXParser.py`.
The script `src/DocumentFactory.py` implements the chunking strategies for the different files type.
Then, in the `/src/stores` folder there is the wrapper for the vector database.

## Usage
To run the application, the user must:
1. Prepare the intermediate representation and the heterogeneous documents.
2. Modify the configuration file according to the requirements.
3. Install the requirements by running `python -m pip install -r requirements.txt`. Using a virtual environment is strongly suggested.
4. Once the requirements has been installed, run the pipeline using `python app.py`
5. When the pipeline finishes the execution, in the `OUTPUT_DIR` folders there will be the execution log and the output JSON.

## Additional Files
Together with the main application, we provide two additional scripts:
- `report.py` to generate the PDF report.
- `show_graph.py` to generate the HTML figure representig the output graph.

## Results
In the `/results` folder we provide the fives evaluations perfomed. Each subfolder details one of them.
- **Case 1**: Goofy Guineapig
- **Case 2**: Smooth Operator
- **Case 3**: Small Sieve
- **Case 4**: Jaguar Tooth
- **Case 5**: COLDSTEEL
- **Case 6**: Umbrella Stand
- **Case 7**: Pygmy Goat
- **Case 8**: Authentic Antics
- **Case 9**: Damascened Peacock
- **Case 10**: Cheeky Chipmunk

Into these subfolders we stored the application log, the output JSON, the PDF report and the HTML visualization and the time and F1 score measures performed using the script `measures.py`.

- `app.log` contains the execution details. From it, it is possible to determine the documents used to generate the responses.
- `execution_scores_XX.log` recorded the precision, recall and F1 score computed during the evaluations. The ground truth we used are in the folder `ground_truths`.
- `execution_times_XX.log` stores the execution times.
- `XX_graph_plot.png` contains the plot of the graph representing the attack.
- `XX_report.pdf` is the automatic report generated from the application output.
- `graph.html` contains the HTML visualization of the graph.
- `LOW_output.json` is the main output of the pipeline, detailing the related attack. This output was generated with a LOW human-interaction level.

Moreover, the STIX used as references are stored in the folder `sample_stix`. In the `documents`folder, we saved the heterogeneous documents used for each experiment.

## Evaluation of the Chunking strategy, the MITRE matching capabilities of the LLMs, and the RAG hyperparameters
We performed different evaluations both to determine the best chunking strategy, the MITRE matching capabilities of the LLMs, and the RAG hyperparameters. The scripts used and the results are available in the `evaluation`. In particular:
- the folder `chunking` contains an example of the chunking strategy implementation as described by  Smith and Troynikov (https://research.trychroma.com/evaluating-chunking)
- the folder `mitre` contains the evaluation of the matching capabilities of the LLMs
- the folder `rag` contains the evaluations to determine the best RAG hyperparameters
