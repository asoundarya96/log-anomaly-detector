from langchain.chat_models import ChatOpenAI


def load_chat_model(openai_api_key, model_name, max_tokens=None, temperature=0.3, modelKwargs=None):
    llm = ChatOpenAI(openai_api_key=openai_api_key, model_name=model_name, temperature=temperature,
                     max_tokens=max_tokens, streaming=True)
    return llm
