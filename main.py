import chromadb
from chromadb.utils import embedding_functions
import csv
import pandas as pd
import streamlit as st

dataset = [
    "I enjoy the experience",
    "Income stream",
    "Beginner",
    "Experienced trader",
    "Long-term investor",
    "Short-term investor",
    "introvert",
    "extrovert",
    "day dreamer",
    "pacticular",
    "Frequently",
    "Periodically",
    "standards",
    "passion",
    "practical people",
    "visionary",
    "Cohesiveness of Thinking",
    "Peaceful Interpersonal Bonds",
    "objective judgement",
    "merit judgement",
    "3-5 years",
    "5 or more years",
    "laid back",
    "earnest and committed",
    "assumes all will be said",
    "prepare your response",
    "Casual",
    "Regular",
    "alluring",
    "trying",
    "compassionate person",
    "even-tempered person",
    "spontaneously",
    "properly planned",
    "Trading only",
    "other means",
    "strike up conversation",
    "hang back for a bit",
    "seldom questioinable",
    "often questionable",
    "emotions",
    "merits",
    "tender then rigid",
    "rigid then tender",
    "conforming to change",
    "strict and well ordered",
    "unrestricted",
    "defined",
    "see others perspectives",
    "see how others bring value",
    "Your mind",
    "Your soul",
    "in group settings",
    "isolated from others",
    "them well maintained",
    "impromtu",
    "constant contact with a lot of friends",
    "deep conversations with few friends",   
    "Manufacturing and Delivery",
    "Planning and Investigation",
    "Forfeit Money",
    "Aquire Money",
    "Unchanging",
    "Initial",
    "Expertise",
    "Intuition",
    "Cautious",
    "Accessible",
    "Non-Formal and Spontaneous",
    "Organized and Planned", 

]
CHROMA_DATA_PATH = "chroma_data/"
EMBED_MODEL = "all-MiniLM-L6-v2"
COLLECTION_NAME = "demo_docs"
file_path = "query.txt"
questions = ""

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

df = pd.read_csv("doc/Rewrite-questions.csv", header=None)

# # df = df.dropna()
questions = df.values.flatten().tolist()

content = ""
st.title("Question and Answer Interaction")

if "question_index" not in st.session_state:
    st.session_state.question_index = 0
    st.session_state.all_answers = ""

question_index = st.session_state.question_index


if question_index < len(questions):
    question = questions[question_index]

    st.header(f"Question {question_index + 1}")
    st.write(question)

    # label1 = "empty"
    # label2 = "empty"
    
    # if dataset[question_index * 2] != "":
    #     label1 = dataset[question_index * 2]
    # if dataset[question_index*2 + 1] != "":
    #     label1 = dataset[question_index * 2+1]
    
    
    answer1 = st.button(
        dataset[question_index * 2], key=f"question_{question_index*2}_option_a"
    )
    answer2 = st.button(
        dataset[question_index * 2+1], key=f"question_{question_index*2+1}_option_a"
    )

    if answer1 or answer2:
        selected_option = (
            dataset[question_index * 2] if answer1 else dataset[question_index * 2 + 1]
        )
        st.session_state.all_answers += f"{selected_option}, "
        st.session_state.question_index += 1
        print("answers:", st.session_state.all_answers)

else:
    st.header("End of the questions")
    query_result = collection.query(
        query_texts=[f"{st.session_state.all_answers}"],
        n_results=1,
    )
    print("", query_result["metadatas"])
    print(query_result.keys())
    print(query_result["metadatas"])

    st.write("Class name:", query_result["metadatas"][0])
