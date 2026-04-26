import os 
from.ollama_client import OLLamaLLM
# from llm.hf_client import HuggingFaceLLM

def get_llm():
    provider = os.getenv("LLM_PROVIDER", "ollama")

    if provider == "ollama":
        return OLLamaLLM()
    # elif provider == "hf":
    # return HuggingFaceLLm()

    raise ValueError("invalid LLM Provider")
    