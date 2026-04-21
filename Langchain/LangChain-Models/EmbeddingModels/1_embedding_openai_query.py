from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
load_dotenv()

embeddings = OpenAIEmbeddings(model = "text-embedding-3-small", dimensions=32)

text = "Hazrat Muhammad s.a.w.w is the last prophet of Allah"

vector = embeddings.embed_query(text)
print(str(vector))