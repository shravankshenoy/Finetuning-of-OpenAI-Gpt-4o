import os
import time
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

client = OpenAI()

# Create an Eval 
# Use Text Similarity Grader
eval_obj = client.evals.create(
    name="Customer Support 1",
    data_source_config={
        "type": "custom",
        "item_schema": {
            "type": "object",
            "properties": {
                "input": {"type": "string"},
                "ground_truth": {"type": "string"},
            },
            "required": ["input", "ground_truth"],
        },
        "include_sample_schema": True,
    },
    testing_criteria=[
        {
            "type": "text_similarity",
            "name": "Check similarity of output to human label",
            "input": "{{ sample.output_text }}",
            "reference": "{{ item.ground_truth }}",
            "pass_threshold": 0.5,
            "evaluation_metric": "bleu"
        }
    ],
)

print(eval_obj)
eval_id = eval_obj.id

# Upload test data 
file = client.files.create(
    file=open("data/eval.jsonl", "rb"),
    purpose="evals"
)

print(file)
file_id = file.id

# Create an eval run on finetuned model
run_ft = client.evals.runs.create(
    eval_id,
    name="Categorization text run",
    data_source={
        "type": "completions",
        "model": "ft:gpt-4o-mini-2024-07-18:personal:custom-fine-tuned-model:BsghFcsc",
        "input_messages": {
            "type": "template",
            "template": [
                {"role": "system", "content": "You are a helpful assistant which acts as FAQ Support Assistant answers to user queries."},
                {"role": "user", "content": "{{ item.input }}"},
            ],
        },
        "source": {"type": "file_id", "id": file_id},
    },
)

print(run_ft)

run_ft_id = run_ft.id

# Create an eval run on base model
run_base = client.evals.runs.create(
    eval_id,
    name="Categorization text run",
    data_source={
        "type": "completions",
        "model": "gpt-4o-mini",
        "input_messages": {
            "type": "template",
            "template": [
                {"role": "system", "content": "You are a helpful assistant which acts as FAQ Support Assistant answers to user queries."},
                {"role": "user", "content": "{{ item.input }}"},
            ],
        },
        "source": {"type": "file_id", "id": file_id},
    },
)

print(run_base)

# Analyze the results
#run = client.evals.runs.retrieve(eval_id, run_id)
#print("###### RESULTS #############")
#print(run)