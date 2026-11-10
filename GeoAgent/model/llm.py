from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langchain_deepseek import ChatDeepSeek
from langchain_openai.chat_models.base import BaseChatOpenAI
import os
import configparser

config = configparser.ConfigParser()
config.read("config.ini")

OPENAI_BASE_URL = os.environ['OPENAI_BASE_URL']
API_KEY = os.environ['API_KEY']

_llm_cache = {}

def create_llm(agent_name: str, temperature: float = 1):
    model_name = config.get("LLM", agent_name)
    if model_name in _llm_cache:
        return _llm_cache[model_name]
    
    if "gpt" in model_name.lower():
        llm = ChatOpenAI(
            model=model_name,
            temperature=1,
            base_url=OPENAI_BASE_URL,
            api_key=API_KEY,
        )
        _llm_cache[model_name] = llm
        return llm
    # elif "deepseek/" in model_name.lower():
    #     llm = ChatDeepSeek(
    #         model=model_name[9:],
    #         api_key=DEEPSEEK_API_KEY,
    #         temperature=1,
    #     )
    #     _llm_cache[model_name] = llm
    #     return llm
    # elif "deepseek-openai/" in model_name.lower():
    #     llm = BaseChatOpenAI(
    #         model=model_name[16:],
    #         base_url="https://api.deepseek.com/v1",
    #         api_key=DEEPSEEK_API_KEY,
    #         temperature=temperature,
    #     )
    #     _llm_cache[model_name] = llm
    #     return llm
    else:
        llm = ChatOpenAI(
            model=model_name,
            temperature=1,
            base_url=OPENAI_BASE_URL,
            api_key=API_KEY,
        )
        _llm_cache[model_name] = llm
        return llm
