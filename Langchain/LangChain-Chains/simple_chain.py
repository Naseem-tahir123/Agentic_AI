# from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

prompt = PromptTemplate(
    template = "Generate five interesting facts about {topic}",
    input_variables = ["topic"]
)

model = ChatGoogleGenerativeAI(model = "gemini-2.5-flash")
parcer = StrOutputParser()


chain = prompt | model | parcer 

result = chain.invoke({"topic":"Gilgit Baltistan"})
print(result)

chain.get_graph().print_ascii()


