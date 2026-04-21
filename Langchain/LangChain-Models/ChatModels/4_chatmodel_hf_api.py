from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    task = "text-generation"
)

chatmodel = ChatHuggingFace(llm=llm)
result = chatmodel.invoke("which city is the capital of Nepal?")
print(result.content)