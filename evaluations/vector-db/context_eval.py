from datasets import load_dataset
from ragas import SingleTurnSample
from ragas.metrics import NonLLMContextPrecisionWithReference
from langchain_community.vectorstores import Chroma, FAISS, LanceDB
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
import asyncio

k = 3

dataset = load_dataset("stepkurniawan/sustainability-methods-wiki", "50_QA")

context_precision = NonLLMContextPrecisionWithReference()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
chunks = text_splitter.create_documents(dataset['train']['contexts'])
db = Chroma.from_documents(chunks, OllamaEmbeddings(model="llama3.1:8b"))

query = dataset['train']['question'][0]
retrieved_docs = db.similarity_search(query, k)

context_precision = NonLLMContextPrecisionWithReference()
sample = SingleTurnSample(
    retrieved_contexts=[doc.page_content for doc in retrieved_docs],
    reference_contexts=[dataset['train']['ground_truths'][0], dataset['train']['contexts'][0]],
)

print(asyncio.run(context_precision.single_turn_ascore(sample)))