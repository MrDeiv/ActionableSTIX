import chromadb
from sklearn.decomposition import PCA
import plotly.express as px
from src.stores.DocumentStore import DocumentStore
import json
import os

CONFIG_FILE = "config.json"
config = json.loads(open(CONFIG_FILE).read())

ds = DocumentStore(
    model_name=config["MODELS"]['SUMMARIZATION']['NAME'],
    collection_name=config["CHROMA_COLLECTION"]['NAME'],
    persist_directory=config["CHROMA_COLLECTION"]['DIR']
)

# ingest documents
#ds.ingest(config["DOCUMENTS_DIR"])

# Get embeddings
db = ds.vector_db
embeddings = db.get(include=['embeddings'])['embeddings']
docs = db.get(include=['embeddings'])['ids']

# Reduce the embedding dimensionality
pca = PCA(n_components=3)
vis_dims = pca.fit_transform(embeddings)# Create an interactive 3D plot
fig = px.scatter_3d(
    x=vis_dims[:, 0],
    y=vis_dims[:, 1],
    z=vis_dims[:, 2],
    text=docs,
    labels={'x': 'PCA Component 1', 'y': 'PCA Component 2', 'z': 'PCA Component 3'}, # Name it like you want
    title='3D PCA of Embeddings' # Name it like you want
)

# Show the plot
fig.write_image("PCA.png")