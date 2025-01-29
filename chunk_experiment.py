from datasets import load_dataset
from langchain_ollama import ChatOllama
from langchain_ollama import OllamaEmbeddings

dataset = load_dataset("stepkurniawan/sustainability-methods-wiki", "50_QA")

MODEL = "llama3.1:8b"
ollama = ChatOllama(model=MODEL)
embeddings = OllamaEmbeddings(model=MODEL)

# {'train': ['contexts', 'summary', 'question', 'ground_truths']}

sample_docs = [
    "Albert Einstein proposed the theory of relativity, which transformed our understanding of time, space, and gravity.",
    "Marie Curie was a physicist and chemist who conducted pioneering research on radioactivity and won two Nobel Prizes.",
    "Isaac Newton formulated the laws of motion and universal gravitation, laying the foundation for classical mechanics.",
    "Charles Darwin introduced the theory of evolution by natural selection in his book 'On the Origin of Species'.",
    "Ada Lovelace is regarded as the first computer programmer for her work on Charles Babbage's early mechanical computer, the Analytical Engine."
]