from fastapi import FastAPI
from RAG import rag_root

app = FastAPI()

@app.get("/")
def read_root():
    return {"message":"The Backend is Live!"}

@app.get("/RAG")
def get_rag():
    return rag_root()
