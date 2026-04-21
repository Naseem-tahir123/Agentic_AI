from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

prompt1 = PromptTemplate(
    template = "Write a detail note on {topic}",
    input_variables=["topic"]
)

prompt2 = PromptTemplate(
    template = "Generate a 5 pointer summary of the following note in simple words \n {note}",
    input_variables=["note"]
)

parser = StrOutputParser()

chain = prompt1 | model| parser | prompt2 | model | parser

result = chain.invoke({"topic": "Democracy"})

print(result)

chain.get_graph().print_ascii()