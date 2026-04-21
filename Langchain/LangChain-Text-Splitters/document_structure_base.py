# markdown splitting
from langchain.text_splitter import RecursiveCharacterTextSplitter, Language
 

text = """
# Project TitleLangChain Text Splitters
This project demonstrates the use of LangChain text splitters to divide large text documents into smaller, manageable chunks. It includes examples of length-based splitting, structure-based splitting, and language-specific code splitting.
## Introduction
LangChain is a powerful framework for developing applications powered by language models. One of its key features is the ability to split large text documents into smaller chunks, which can be useful for various applications such as chatbots, summarization, and question-answering.
## Features
- Length-Based Splitting: Divides text based on specified character limits.
- Structure-Based Splitting: Splits text based on logical structures such as paragraphs and headings.
- Language-Specific Code Splitting: Tailored splitting for programming languages like Python.
## Usage
To use the text splitters, simply create an instance of the desired splitter class and call the appropriate method to split your text.
"""

splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.MARKDOWN,
    chunk_size=400,
    chunk_overlap=0
)

texts = splitter.split_text(text)
print(len(texts))
print(texts[1])