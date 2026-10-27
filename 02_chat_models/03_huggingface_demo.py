from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
import os

load_dotenv()

# Verify the token is loaded (optional, for debugging)
# print("Token loaded:", bool(os.getenv("HUGGINGFACEHUB_API_TOKEN")))

# Use a model with wide serverless provider support
llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)

result = model.invoke("Hello, how are you?")
print(result.content)
