import os
import pandas as pd
import json
from openai import OpenAI
from dotenv import load_dotenv


load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


def csv_to_jsonl():
    df = pd.read_csv('Customer-Support.csv')
    df = df.head(15)
    f = open("data-15.jsonl", "w")

    for _, row in df.iterrows():
        json_line = json.dumps({"messages":[
            {"role":"system", "content":"You are a helpful assistant"},
            {"role":"user", "content":row['query']},
            {"role":"assistant", "content":row['response']}
        ]})
        f.write(json_line + '\n')

    f.close()

# csv_to_jsonl()

client = OpenAI(api_key=OPENAI_API_KEY)

output_file = 'data-15.jsonl'
uploaded_file = client.files.create(
    file=open(output_file, "rb"),
    purpose="fine-tune"
)
print(f"File uploaded successfully. File ID: {uploaded_file.id}")

fine_tune_job = client.fine_tuning.jobs.create(
    training_file=uploaded_file.id,
    suffix="custom-fine-tuned-model",
    model="gpt-4o-mini-2024-07-18"  # Adjust the model as required
)
print(f"Fine-tuning job started. Job ID: {fine_tune_job.id}")

# List fine-tuning jobs
jobs = client.fine_tuning.jobs.list(limit=10)
print("Recent fine-tuning jobs:", jobs)

# Retrieve job details
job_details = client.fine_tuning.jobs.retrieve(fine_tune_job.id)
print("Fine-tuning job details:", job_details)

# Retrieve after model is trained to get the model name
job_details = client.fine_tuning.jobs.retrieve(fine_tune_job.id)
print("Fine-tuning job details:", job_details)



