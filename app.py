from openai import OpenAI
from dotenv import load_dotenv
import os
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

client = OpenAI()

# Call to the fine-tuned model
completion = client.chat.completions.create(
    model="ft:gpt-4o-mini-2024-07-18:personal:custom-fine-tuned-model:BsghFcsc",
    messages=[
        {"role": "system", "content": "You are a helpful assistant which acts as FAQ Support Assistant for the TMLC Guided Projects in Generative AI Program and answer to user queries."},
        {"role": "user", "content": "How to cancel subscription?"}
    ]
)
print("Fine-tuned model response:", completion.choices[0].message.content)