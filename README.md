# Finetuning Open AI GPT-4o 

## Problem Statement
The goal is to build a bot which can respond to customer queries and complaints for an e-commerce website


## Approach
1. Dataset Preparation
    * Downloaded Customer Support response dataset (`kaludi/Customer-Support-Responses`) from Huggingface
    * Formatted the data to use chat completions format and use the jsonl file format
    * Prepared evaluation data in jsonl format

2. Finetuning Gpt-4o 
    * Ran fine tuning job to finetune `gpt-4o-mini-2024-07-18` on above data

3. Evaluating fine tuned model
    * Ran evaluation on fine-tuned model with the evaluation data. Used the following metrics
        * Text similarity using BLEU (bilingual evaluation understudy)


## Results

| Evaluation Metric                         | Gpt-4o-mini| Gpt-4o-mini-finetuned|
|-------------------------------------------|------------|----------------------|
| Text Similarity using BLEU                | 0.0        | 0.66                 |



## Next Steps
- [ ] Use more metrics for evaluation
- [ ] Increase size of evaluation data
- [ ] Improve eval results using dataset refinement and context enhancement (updating system instruction)