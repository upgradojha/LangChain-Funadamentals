from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI(model='gpt-5-nano')
result =model.invoke('who is the priminister of india')

# print(result)
print(result.content)

