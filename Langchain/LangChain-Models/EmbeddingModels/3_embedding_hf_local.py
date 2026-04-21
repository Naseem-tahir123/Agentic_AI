from langchain_huggingface import HuggingFaceEmbeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

text = "I am from gilgit baltistan"

vector = embeddings.embed_query(text)

print(str(vector))