# from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import TextLoader
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

prompt = PromptTemplate(
    template= "Write the summary of the following \n {text}",
    input_variables=["text"]
)

parcer = StrOutputParser()


loader = TextLoader("democracy.txt", encoding="utf8")

docs =  loader.load()


chain = prompt | model | parcer

result = chain.invoke({"text": docs[0].page_content})

print(f"The summary of the poem is: \n {result}")

# print(len(docs))
# # print(docs[0])
# print(docs[0].page_content[:500])
# print(docs[0].metadata)