from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

load_dotenv()

embedding = OpenAIEmbeddings(model='text-embedding-3-small', dimensions=30)

documents = [
    'Narendra modi is prime minister of india',
    'donal trump is a fucking idot',
    'India is great country of the Entitiled peoples'
]
query = 'Tell me about modi'
documents_embedding= embedding.embed_documents(documents)
query_embedding = embedding.embed_query(query)

scores = cosine_similarity(np.array([query_embedding]),np.array(documents_embedding))[0]

ranked = sorted(enumerate(scores), key=lambda x:x[1], reverse=True)

idx, rank = ranked[0]
print('query is :', query)
print('ans is :', documents[idx])
