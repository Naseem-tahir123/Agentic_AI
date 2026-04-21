from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
load_dotenv()


embeddings = OpenAIEmbeddings(model= "text-embedding-3-large", dimensions=32)

docs = [
    "My name is Naseem Tahir",
    "I am an AI Engineer",
    "I belong to Gilgit Baltistan"
]

vector = embeddings.embed_documents(docs)
print(str(vector))
