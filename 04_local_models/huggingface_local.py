from langchain_huggingface import HuggingFacePipeline, ChatHuggingFace
import torch

# The official model ID for the 0.5B instruction-tuned Qwen model
model_id = "Qwen/Qwen2.5-0.5B-Instruct"

llm = HuggingFacePipeline.from_model_id(
    model_id=model_id,
    task="text-generation",
    model_kwargs={
        # For CPU, float32 is often the most stable and performant choice
        "dtype": torch.float32,
        "device_map": "cpu",  # Force CPU to prevent slow disk offloading
        "trust_remote_code": True,
    },
    pipeline_kwargs={
        "max_new_tokens": 256,  # Lower this (e.g., 128) for faster tests
        "temperature": 0.7,
        "do_sample": True,
    },
)

model = ChatHuggingFace(llm=llm)

while True:
    print("press 4 for exit")
    query = input("Enter the your Query: ")
    result = model.invoke(query)
    print(result.content)
    if query =="1":
        break
