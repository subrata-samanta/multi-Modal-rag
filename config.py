import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    GOOGLE_CREDENTIALS_PATH = os.getenv("GOOGLE_CREDENTIALS_PATH", "key.json")
    
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200
    COLLECTION_NAME = "multimodal_rag"
    PERSIST_DIRECTORY = "./chroma_db"
