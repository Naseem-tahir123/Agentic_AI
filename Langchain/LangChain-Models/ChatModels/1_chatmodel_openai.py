from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
load_dotenv()
chat = ChatOpenAI(model="gpt-4o-mini")
result = chat.invoke("which is the most attractive place in gilgit baltistan?")
print(result.content)