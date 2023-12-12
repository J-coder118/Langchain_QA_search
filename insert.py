import chromadb
from chromadb.utils import embedding_functions
import csv
import pandas as pd
import streamlit as st


CHROMA_DATA_PATH = "chroma_data/"
EMBED_MODEL = "all-MiniLM-L6-v2"
COLLECTION_NAME = "demo_docs"

# print("kdkdkdkdk")
client = chromadb.PersistentClient(path=CHROMA_DATA_PATH)

embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name=EMBED_MODEL
)

# collection = client.get_or_create_collection(
collection = client.create_collection(
    name=COLLECTION_NAME,
    embedding_function=embedding_func,
    metadata={"hnsw:space": "cosine"},
)

# df = pd.read_csv('doc/new_dataset.csv')
df = pd.read_csv('doc/merged_file.csv')
df = df.dropna()

questions = ['"' + str(value) + '"' for value in df['questions'].values]
className = ['"' + str(value) + '"' for value in df['className'].values]

collection.add(
    documents=questions,
    ids=[f"id{i}" for i in range(len(questions))],
    metadatas=[{"className": cn} for cn in className]
)

    