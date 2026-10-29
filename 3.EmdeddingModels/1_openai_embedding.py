from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

embedding = OpenAIEmbeddings(model='text-embedding-3-small', dimensions=4)

result = embedding.embed_query("Hii")
result2 = embedding.embed_query("Hello")
print(result)
print(result2)


