from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv
load_dotenv()

chatmodel = ChatAnthropic(model="claude-2")
result = chatmodel.invoke("Tell me about the area of Gilgit Baltistan?")
print(result.content)