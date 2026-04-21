from langchain_openai import OpenAIEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from dotenv import load_dotenv
load_dotenv()

embeddings = OpenAIEmbeddings()
docs = [
    "A gem of Gilgit-Baltistan, Hunza Valley enchants with terraced fields, lush greenery, and panoramic views of snow-capped peaks like Rakaposhi and Ultar.",
    "Nestled at the base of Nanga Parbat, this alpine meadow offers tranquil natural beauty, ideal for camping and disconnecting from urban life.",
    "Formed by a landslide in 2010, the turquoise waters of Attabad Lake are perfect for boating and are framed by dramatic mountain scenery.",
    "One of the highest paved international border crossings in the world, Khunjerab Pass gives breathtaking views of Karakoram peaks and wildlife.",
    "Located in Kharmang Valley near Skardu, Manthokha Waterfall cascades beautifully amidst lush pastures and rugged mountain terrain."
]
 

query = "tell me about  Fairy Meadows?"

docs_embeddings = embeddings.embed_documents(docs)
query_embeddings = embeddings.embed_query(query)

scores = cosine_similarity([query_embeddings], docs_embeddings)[0]
cleaned_scores = [float(i) for i in scores]

index, score = sorted(list(enumerate(cleaned_scores)), key = lambda x: x[1])[-1]

print(f"Question: {query}")
print(f"Answer: {docs[index]}")
print(f"similarity score is: {score}")

