import requests

class OLLamaLLM:
    def __init__(self, model = 'llama3'):
        self.model = model
    
    def generate(self, prompt: str) -> str:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json = {
                "model" : self.model,
                "prompt" : prompt,
                "stream" : False
            }
        )

        return response.json()["response"]