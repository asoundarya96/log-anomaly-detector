from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import TextLoader
from langchain.chains.summarize import load_summarize_chain
import os

from dotenv import load_dotenv
load_dotenv()


prompt_template = """Convert given set of linux log lines to meaningful summaries with date information:
"{logs}"
CONCISE SUMMARY:"""
prompt = PromptTemplate.from_template(prompt_template)

def summarize(doc_path):
    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo-16k")
    llm_chain = LLMChain(llm=llm, prompt=prompt)

    loader = TextLoader(doc_path)
    docs = loader.load()

    stuff_chain = StuffDocumentsChain(
        llm_chain=llm_chain, document_variable_name="logs"
    )

    return stuff_chain.run(docs)

directory_path = "test_data/linux_logs/date_wise_logs"
summary_directory_path = "test_data/linux_logs/summaries"
june_logs = "june_logs.log"

for filename in os.listdir(directory_path):
    print("Converting log files to summaries......")
    file_path = os.path.join(directory_path, filename)
    summary = summarize(file_path)
    with open(summary_directory_path+'/'+june_logs,'a') as fobj:
        fobj.write(summary)
        fobj.write('\n')
print("Completed!")

    
