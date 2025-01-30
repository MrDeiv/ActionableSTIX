"""
This script evaluates the best chunk size for the document store
"""

from datasets import load_dataset
from langchain_ollama import ChatOllama
from langchain_ollama import OllamaEmbeddings
from ragas.metrics import LLMContextRecall, Faithfulness, FactualCorrectness
from ragas import evaluate, EvaluationDataset
from ragas.metrics import ContextPrecision, ContextRecall
from ragas.llms import LangchainLLMWrapper

from src.DocumentStore import DocumentStore
from src.ChatModel import ChatModel

MODEL = "llama3.1:8b"
START = 100
INCREMENT = 100
MAX_CHUNK_SIZE = 2000

d = load_dataset("stepkurniawan/sustainability-methods-wiki", "50_QA")

# {'train': ['contexts', 'summary', 'question', 'ground_truths']}
queries = d['train']['question']
expected_responses = d['train']['ground_truths']

dataset = []

retriever = DocumentStore(model=MODEL).retriever
qa_chain = ChatModel(model_name=MODEL, retriever=retriever)

for query, reference in zip(queries, expected_responses):
    relevant_docs = retriever.invoke(query)
    response = qa_chain.invoke(query)
    dataset.append(
        {
            "user_input": query,
            "retrieved_contexts": [rdoc.page_content for rdoc in relevant_docs],
            "response": response,
            "reference": reference,
        }
    )

evaluation_dataset = EvaluationDataset.from_list(dataset)

evaluator_llm = LangchainLLMWrapper(
    ChatOllama(model=MODEL)
)

result = evaluate(
    dataset=evaluation_dataset,
    metrics=[LLMContextRecall(), Faithfulness(), FactualCorrectness()],
    llm=evaluator_llm,
)