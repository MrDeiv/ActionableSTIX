from transformers import (AutoModelForCausalLM, AutoTokenizer, pipeline)
from src.DocumentFactory import DocumentFactory
from langchain_huggingface import HuggingFaceEndpoint, HuggingFacePipeline, ChatHuggingFace
import os
import operator
from typing import List, Literal, TypedDict

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig
from langgraph.constants import Send
from langgraph.graph import END, START, StateGraph
from langchain.text_splitter import NLTKTextSplitter
from langchain_core.documents import Document
import asyncio
from langchain_community.document_loaders import WebBaseLoader
from langchain_ollama import ChatOllama
from tqdm import tqdm

class State(TypedDict):
        contents: List[str]
        index: int
        summary: str


# Here we implement logic to either exit the application or refine the summary.
def should_refine(state: State) -> Literal["refine_summary", END]:
    if state["index"] >= len(state["contents"]):
        return END
    else:
        return "refine_summary"

async def main():
    model_name = "microsoft/Phi-3.5-mini-instruct"
    model = AutoModelForCausalLM.from_pretrained(model_name) 
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    hf_p = HuggingFacePipeline(
        pipeline=pipeline(
            "text-generation",
            model=model, 
            tokenizer=tokenizer, 
            device="cpu", 
            trust_remote_code=True, 
            temperature=0.5, 
            do_sample=True,
            max_new_tokens=2048), 
        verbose=True)
    #chat_summary = ChatHuggingFace(llm=hf_p, verbose=True)
    chat_summary = ChatOllama(model='llama3.1:8b', temperature=0.5, verbose=True)

    folder = "./documents/other/"

    text = """
    In December 1903 the Royal Swedish Academy of Sciences awarded Pierre Curie, Marie Curie, and Henri Becquerel the Nobel Prize in Physics,[47] "in recognition of the extraordinary services they have rendered by their joint researches on the radiation phenomena discovered by Professor Henri Becquerel."[23] At first the committee had intended to honour only Pierre Curie and Henri Becquerel, but a committee member and advocate for women scientists, Swedish mathematician Magnus Gösta Mittag-Leffler, alerted Pierre to the situation, and after his complaint, Marie's name was added to the nomination.[48] Marie Curie was the first woman to be awarded a Nobel Prize.[23]

Curie and her husband declined to go to Stockholm to receive the prize in person; they were too busy with their work, and Pierre Curie, who disliked public ceremonies, was feeling increasingly ill.[46][48] As Nobel laureates were required to deliver a lecture, the Curies finally undertook the trip in 1905.[48] The award money allowed the Curies to hire their first laboratory assistant.[48] Following the award of the Nobel Prize, and galvanised by an offer from the University of Geneva, which offered Pierre Curie a position, the University of Paris gave him a professorship and the chair of physics, although the Curies still did not have a proper laboratory.[23][43][44] Upon Pierre Curie's complaint, the University of Paris relented and agreed to furnish a new laboratory, but it would not be ready until 1906.
    """

    # initial summary
    summarize_prompt = ChatPromptTemplate(
        [
            ("human", "Write a summary of the following: {context}"),
        ]
    )
    initial_summary_chain = summarize_prompt | chat_summary | StrOutputParser()

    # Refining the summary with new docs
    refine_template = """
    Produce a final summary.

    Existing summary up to this point:
    {existing_answer}

    New context:
    ------------
    {context}
    ------------

    Given the new context, refine the original summary.
    """
    refine_prompt = ChatPromptTemplate([("human", refine_template)])

    refine_summary_chain = refine_prompt | chat_summary | StrOutputParser()

    # We define functions for each node, including a node that generates the initial summary:
    async def generate_initial_summary(state: State, config: RunnableConfig):
        summary = await initial_summary_chain.ainvoke(
            state["contents"][0],
            config,
        )
        return {"summary": summary, "index": 1}


    # And a node that refines the summary based on the next document
    async def refine_summary(state: State, config: RunnableConfig):
        content = state["contents"][state["index"]]
        summary = await refine_summary_chain.ainvoke(
            {"existing_answer": state["summary"], "context": content},
            config,
        )
        return {"summary": summary, "index": state["index"] + 1}


    # We will define the state of the graph to hold the document
    # contents and summary. We also include an index to keep track
    # of our position in the sequence of documents.


    graph = StateGraph(State)
    graph.add_node("generate_initial_summary", generate_initial_summary)
    graph.add_node("refine_summary", refine_summary)

    graph.add_edge(START, "generate_initial_summary")
    graph.add_conditional_edges("generate_initial_summary", should_refine)
    graph.add_conditional_edges("refine_summary", should_refine)
    app = graph.compile()

    # We will now load the documents and run the application
    text_splitter = NLTKTextSplitter(chunk_size=1000, chunk_overlap=1000*0.3)
    #chunks = text_splitter.split_text(text)
    #documents = [Document(page_content=chunk) for chunk in chunks]

    loader = WebBaseLoader("https://en.wikipedia.org/wiki/Marie_Curie")
    docs = loader.load()

    documents = []

    progress = tqdm(docs, desc="Processing documents")
    for doc in docs:
        progress.update(1)
        chunks = text_splitter.split_text(doc.page_content)
        d = text_splitter.create_documents(chunks)
        documents.extend(d)
    progress.close()

    res = await app.ainvoke(
        {"contents": [doc.page_content for doc in documents]},
        RunnableConfig(),
        stream_mode="values"
    )

    summary:str = res["summary"]
    
    # get only the assistant summary
    limit = "<|assistant|>"
    # get last index of limit
    index = summary.rfind(limit)
    # get the assistant summary
    summary = summary[index + len(limit):].strip().replace("\n", " ")

    with open("summary.txt", "w") as f:
        f.write(summary)

    
if __name__ == "__main__":
    asyncio.run(main())

