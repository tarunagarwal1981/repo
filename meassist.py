import sys
import pysqlite3
sys.modules['sqlite3'] = pysqlite3
import chromadb
from chromadb.config import Settings
import subprocess
import tempfile
import os
import streamlit as st
import requests
from io import BytesIO
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.chains import ConversationalRetrievalChain

# Constants and API Keys
OPENAI_API_KEY = "open_api_key"
VECTOR_DB_DIRECTORY = "vectordb"
GPT_MODEL_NAME = 'gpt-3.5-turbo'
CHUNK_SIZE = 700
CHUNK_OVERLAP = 50
GITHUB_REPO_URL = "https://api.github.com/repos/tarunagarwal1981/repo/contents/?ref=tarunagarwal1981-knowledgehub"

# Function Definitions
# ... (keep the existing function definitions)

# Vector Database Initialization
vector_database = None

def initialize_vector_database():
    global vector_database
    if not vector_database:
        vector_database = Chroma(persist_directory=VECTOR_DB_DIRECTORY, embedding_function=create_embeddings(OPENAI_API_KEY), client_settings=Settings())
    return vector_database

# Main Execution Flow
def main():
    model_trained = check_model_trained_status()
    st.title("Main Engine Support")

    if not model_trained:
        st.warning("Model is not trained. Please train the model first.")
        if st.button("Train Model"):
            train_model()
            st.experimental_rerun()
    else:
        vector_db = initialize_vector_database()
        chat_model = initialize_chat_model(OPENAI_API_KEY, GPT_MODEL_NAME)
        qa_chain = create_retrieval_qa_chain(chat_model, vector_db)

        question = st.text_input("Ask a question")

        if question:
            answer = ask_question_and_get_answer(qa_chain, question)
            st.write(f"Answer: {answer}")

        if st.button("Retrain Model"):
           retrain_model()

if __name__ == "__main__":
    main()