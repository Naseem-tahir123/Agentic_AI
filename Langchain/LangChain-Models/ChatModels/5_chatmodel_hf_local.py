from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from dotenv import load_dotenv
import os

# use this line if you don't have enough memory in C drive
os.environ["HF_HOME"] = "D:/HFModels"  # Change this to your desired path
load_dotenv()

llm = HuggingFacePipeline.from_model_id(
    model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task = "text-generation",
    pipeline_kwargs=dict(
        temperature = 0.2,
        max_new_tokens = 512,        
)
    
)

model = ChatHuggingFace(llm=llm)

result = model.invoke("In which year more flood occurred in Pakistan?")
print(result.content)