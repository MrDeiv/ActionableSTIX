# A Pipeline for Enriching Threat Intelligence with LLMs

## Abstract
Effective cyber threat intelligence hinges on data collection and the ability to swiftly contextualize and act on that data. While STIX (Structured Threat Information Expression) provides a standardized format for sharing threat indicators, it often lacks the actionability analysts require. This work presents a novel pipeline that enhances STIX files through external document enrichment powered by Large Language Models (LLMs).
The pipeline comprises two phases: 1) ingesting structured threat data and relevant additional content and 2) producing enriched outputs. LLMs extract and correlate information, generating actionable intelligence for analyst consumption. The resulting system bridges the gap between raw threat data and informed, rapid decision-making.

## Pipeline Overview
![Pipeline Overview](docs/schema_new.jpg)

In this section, we match the components described in the schema above with the code's components provided in this repository.

The STIX file and the additional documents, previously retrieved manually, must be inserted in the proper directories.

Users should verify the configuration file in `config/config.json` before executing the pipeline.

### Configuration File
This file allows users to customize their implementation, configuring the parameters as follows:
- `STIX_FILE`: the path of the STIX file.
- `DOCUMENTS_DIR`: the directory containing the additional files.
- `OUTPUT_DIR`: the output directory. It will contains the log file and the output JSON.
- `OUTPUT_FILE`: the name of the output file. Note that the `SELECTED_INTERACTION_LEVEL` will be inserted at the start of the filename.
- `MODELS`: the models to use in the pipeline. In particular, it requires the name of the `TEXT_GENERATION` and the `SENTENCE_TRANSFORMER` models.
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
To parse the STIX file and extract the interesting objects we implemented the class `src/STIXParser.py`.
The script `src/DocumentFactory.py` implements the chunking strategies for the different files type.
Then, in the `/src/stores` folder there is the wrapper for the vector database.

## Usage
To run the application, the user must:
1. Prepare the STIX file and the additional resources.
2. Modify the configuration file according to the requirements.
3. Install the requirements by running `python -m pip install -r requirements.txt`. using a virtual environment is strongly suggested.
4. Once the requirements has been installed, run the pipeline using `python app.py`
5. When the pipeline finishes the execution, in the `OUTPUT_DIR` folders there will be the execution log and the output JSON.

## Additional Files
Together with the main application, we provide two additional scripts:
- `report.py` to generate the PDF report.
- `show_graph.py` to generate the HTML figure representig the output graph.

## Results
In the `/results` folder we provide the fives evaluations perfomed. Each subfolder details one of them.

Into these subfolders we stored the application log, the output JSON, the PDF report and the HTML visualization and the time and F1 score measures performed using the script `measures.py`.