from langchain_openai import OpenAIEmbeddings

from dotenv import load_dotenv

load_dotenv()

document = [
    'text 1',
    'text 2',
    'text 3'
]

embeding = OpenAIEmbeddings(model='text-embedding-3-small', dimensions=12)

result = embeding.embed_documents(document)
for res in result:
    print(res)
    print()
