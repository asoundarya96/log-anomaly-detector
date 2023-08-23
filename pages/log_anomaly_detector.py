import streamlit as st
from src.components.chat import display_chat
from src.modules.models import load_chat_model
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.document_loaders import DirectoryLoader
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
import os
from langchain import PromptTemplate
from langchain import FewShotPromptTemplate

load_dotenv()

template = """Use below context to answer the following questions.
Consider you are a SRE, who can monitor and observe application using given context logs.
if user asks about state of system, list down all the unique events happened during that time period in logs.
You need to print only unique error messages and explain the anomaly.
Also indicate if it a suspicious attempt.
Context: {context}
Question: {question}
Answer:"""
QA_CHAIN_PROMPT = PromptTemplate.from_template(template)

def load_docs(directory):
    loader = DirectoryLoader(directory)
    documents = loader.load()
    return documents


def split_docs(documents, chunk_size=1000, chunk_overlap=20):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    docs = text_splitter.split_documents(documents)
    return docs

def load_log_data(directory):
    documents = load_docs(directory)
    docs = split_docs(documents)
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    db = Chroma.from_documents(docs, embeddings)
    return db


def load_qa_chain(llm,db,prompt_template):
    chain_type_kwargs= {"prompt": prompt_template}
    return RetrievalQA.from_chain_type(llm, retriever=db.as_retriever(), chain_type_kwargs=chain_type_kwargs)


def log_agent(query):
    llm = load_chat_model(os.getenv('OPENAI_API_KEY'), "gpt-3.5-turbo")
    db = load_log_data("test_data/linux_logs")
    chain = load_qa_chain(llm, db, QA_CHAIN_PROMPT)
    answer = chain({"query": query})
    return answer['result']



def main():
    st.set_page_config(page_title="Automon")
    st.header('Log Anamoly Detector')
    display_chat(log_agent)

if __name__ == "__main__":
    main()