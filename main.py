import chromadb
from chromadb.utils import embedding_functions
import csv
import pandas as pd
import streamlit as st

CHROMA_DATA_PATH = "chroma_data/"
EMBED_MODEL = "all-MiniLM-L6-v2"
COLLECTION_NAME = "demo_docs"

client = chromadb.PersistentClient(path=CHROMA_DATA_PATH)

embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name=EMBED_MODEL
)

collection = client.get_or_create_collection(
    name=COLLECTION_NAME,
    embedding_function=embedding_func,
    metadata={"hnsw:space": "cosine"},
)

# questions = []
# className = []

# df = pd.read_csv('doc/merged_file.csv')

# df = df.dropna()

# questions = ['"' + str(value) + '",' for value in df['questions'].values]
# className = ['"' + str(value) + '",' for value in df['className'].values]

# print(questions)
# print(className)

# collection.add(
#     documents=questions,
#     ids=[f"id{i}" for i in range(len(questions))],
#     metadatas=[{"className": cn} for cn in className]
# )


st.title("chat")
st.caption("Please ask what you want.")

question = st.text_input("Question")

if st.button("Get Answer"):
    query_result = collection.query(
        query_texts=[f"{question}"],
        n_results=1,
    )

    # print(query_result.keys())
    # print(query_result["metadatas"])

    st.write("Class name:", query_result["metadatas"][0][0]["className"], query_result["distances"])

