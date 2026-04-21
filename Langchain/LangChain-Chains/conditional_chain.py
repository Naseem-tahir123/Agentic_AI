from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from langchain.schema.runnable import RunnableParallel, RunnableBranch, RunnableLambda
from pydantic import BaseModel, Field
from typing import Literal
from dotenv import load_dotenv

load_dotenv()


class SentimentResponse(BaseModel):
    sentiment : Literal['Positive', 'Negative', 'Neutral'] = Field(description="The sentiment of the text")

parcer = StrOutputParser()
parcer1 = PydanticOutputParser(pydantic_object=SentimentResponse)
model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

prompt = PromptTemplate(
    template="Find the sentiment of the given text: \n {text} \n {format_instruction}, ",
    input_variables=["text"],
    partial_variables={"format_instruction": parcer1.get_format_instructions()}
)
 
classifier_chain = prompt | model | parcer1

prompt2 = PromptTemplate(
    template="Give the appropriate response for this positive feedback in one line: \n {feedback}",
    input_variables=["feedback"]
)

prompt3 = PromptTemplate(
    template="Give the appropriate response for this negative feedback in one line: \n {feedback}",
    input_variables=["feedback"]
)

branch_chain = RunnableBranch(
    
        (lambda x: x.sentiment == "Positive", prompt2 | model | parcer),
        (lambda x: x.sentiment == "Negative", prompt3 | model | parcer),
        RunnableLambda(lambda x: "No special response needed for neutral feedback")
    
)
chain = classifier_chain | branch_chain

result =  chain.invoke({"text": "The system perfromance is very bad!"})
print(result)

chain.get_graph().print_ascii()



