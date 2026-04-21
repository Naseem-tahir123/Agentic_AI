from langchain.text_splitter import RecursiveCharacterTextSplitter, Language

# python_code_splitter
# it is an extension of RecursiveCharacterTextSplitter for splitting python code files into smaller chunks based on logical code structures..PYTHON



text = """
class Phone:
    def __init__(self, brand, model, price):
        self.brand = brand
        self.model = model
        self.price = price

    def make_call(self, number):
        return f"Calling {number} from {self.brand} {self.model}"

    def send_message(self, number, message):
        return f"Sending message to {number}: {message}"

# Example usage
phone = Phone("Apple", "iPhone 13", 999)
print(phone.make_call("123-456-7890"))
print(phone.send_message("123-456-7890", "Hello!"))

"""

splitter = RecursiveCharacterTextSplitter.from_language(
    language = Language.PYTHON,
    chunk_size = 400,
    chunk_overlap = 0
)

texts = splitter.split_text(text)

print(len(texts))
print(texts[1])