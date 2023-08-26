import os

from langchain import PromptTemplate, FewShotPromptTemplate
from langchain.chains import RetrievalQA
from langchain.document_loaders import DirectoryLoader
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma

from src.modules.models import load_chat_model

chain = None

QA_CHAIN_PROMPT = PromptTemplate(input_variables=["question", "answer"], template="""
Question: {question} \n
{answer}
""")

examples = [
    {
        "question": "Identify the anomalies on Jun 20",
        "answer": '''
            1. Abnormal exit has occured 1 times
            2. Authentication failures has occured 10 times. This looks like suspicious activity leading towards unauthorized access
        '''
    }
]

prompt = FewShotPromptTemplate(
    examples=examples,
    example_prompt=QA_CHAIN_PROMPT,
    input_variables=["question", "context"],
    prefix='''
    Use below context to answer the following questions.
    {context}
    
    Consider you are a SRE, who can monitor and observe application using given context logs.
    if user asks about state of system, list down all the unique events happened during that time period in logs. 
    ''',
    suffix='''
    Question: {question} \n
    '''
)


def load_docs(directory):
    loader = DirectoryLoader(directory)
    documents = loader.load()
    return documents


def split_docs(documents, chunk_size=1000, chunk_overlap=20):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    docs = text_splitter.split_documents(documents)
    return docs


def get_answer(query):
    global chain
    if chain is None:
        llm = load_chat_model(os.getenv('OPENAI_API_KEY'), "gpt-3.5-turbo")
        chain_type_kwargs = {"prompt": prompt}
        chain = RetrievalQA.from_chain_type(llm, retriever=load_log_data().as_retriever(),
                                            chain_type_kwargs=chain_type_kwargs)
    answer = chain({"query": query})
    return answer


def load_log_data():
    documents = load_docs("test_data/linux_logs")
    docs = split_docs(documents)
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    return Chroma.from_documents(docs, embeddings)
