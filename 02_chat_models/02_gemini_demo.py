from dotenv import load_dotenv

load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI

model = ChatGoogleGenerativeAI(
    model="gemini-flash-lite-latest"  )


result = model.invoke("In single response tell me the about the capital of INDIA in one para")

print(result)
