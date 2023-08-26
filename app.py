import streamlit as st
from dotenv import load_dotenv

from src.components.chat import display_chat
from src.modules.db import get_answer

load_dotenv()


def log_agent(query):
    answer = get_answer(query)
    return answer['result']


def main():
    st.set_page_config(page_title="Automon")
    st.header('Log Anomaly Detector')
    display_chat(log_agent)


if __name__ == "__main__":
    main()
    # while True:
    #     print("Enter your question")
    #     a = input()
    #     print(get_answer(a)['result'])
