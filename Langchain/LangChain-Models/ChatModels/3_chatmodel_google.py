from langchain_google_genai import ChatGoogleGenerativeAI

from dotenv import load_dotenv
load_dotenv()

chatmodel = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature = 0.1, max_completion_tokens = 512)
answer = chatmodel.invoke("In which year the famous Chaqchan Mosque in khaplu was built?")
print(answer.content)