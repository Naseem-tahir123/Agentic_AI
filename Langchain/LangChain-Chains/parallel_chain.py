from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableParallel
from dotenv import load_dotenv
load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",

    task= "text generation",
)

model1 = ChatHuggingFace(llm = llm)
model2 = ChatGoogleGenerativeAI(model = "gemini-2.5-flash")

prompt1 = PromptTemplate(
    template = "Write interactive note on the topic of {topic}.",
    input_variables=["topic"]
)
prompt2 = PromptTemplate(
    template="Generate five quiz questions with four options on the topic of {topic}.",
    input_variables=["topic"],
)
prompt3 = PromptTemplate(
    template="Combine the following notes and quiz into a single study guide: \n Notes: {notes} \n Quiz: {quiz}" ,
    input_variables=["notes", "quiz"],
)

parcer = StrOutputParser()

parallel_chain = RunnableParallel(
    {
        "notes" : prompt1 | model1 | parcer,
        "quiz" : prompt2 | model2 | parcer
    }
)

merge_chain = prompt3 | model2 | parcer

chain = parallel_chain | merge_chain

result = chain.invoke({"topic" : "Tools to build Agent (n8n)"})

print(result)
chain.get_graph().print_ascii()

