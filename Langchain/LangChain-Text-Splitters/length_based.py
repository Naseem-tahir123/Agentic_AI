from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("Naseem Euro pass cv.pdf")
doc  = loader.load()
 
# text = """LangChain is a framework for developing applications powered by language models. It can be used for chatbots, Generative Question-Answering (GQA), summarization, and much more. LangChain provides a standard interface for all LLMs, as well as a suite of tools to work with them. It includes modules for prompt management, memory management, and integration with various data sources. It also supports chains, which are sequences of calls to LLMs or other utilities, allowing for complex workflows. LangChain is designed to be modular and extensible, making it easy to customize and build upon. Whether you're building a simple chatbot or a complex AI application, LangChain provides the tools and infrastructure to help you succeed."""

splitter = CharacterTextSplitter(
    chunk_size = 100,
    chunk_overlap  = 20,
    separator=""
)


texts = splitter.split_documents(doc)

print(texts[0].page_content)
# for i in doc:
#     print(f'"{i}"\n')
