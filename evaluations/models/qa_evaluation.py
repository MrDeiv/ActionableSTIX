from datasets import load_dataset
from transformers import (AutoModelForQuestionAnswering, AutoTokenizer, QuestionAnsweringPipeline, pipeline)
import evaluate
import json
import numpy as np

ds = load_dataset("stepkurniawan/sustainability-methods-wiki", "50_QA")['train']
bert_score = evaluate.load("bertscore")

models = [
    'deepset/roberta-base-squad2',
    'distilbert-base-cased-distilled-squad',
    'distilbert/distilbert-base-uncased-distilled-squad',
    'deepset/bert-large-uncased-whole-word-masking-squad2',
    'google-bert/bert-large-uncased-whole-word-masking-finetuned-squad'
]

answers = []
ground_truths = []
results = {}
for model_name in models:
    model = AutoModelForQuestionAnswering.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    agent = pipeline("question-answering", model=model, tokenizer=tokenizer)
    for example in ds:
        question = example['question']
        context = example['contexts']
        answer = agent(question=question, context=context)['answer']
        answers.append(answer)
        ground_truths.append(example['ground_truths'])

    res = bert_score.compute(
        predictions=answers, 
        references=ground_truths, 
        lang="en")
    
    results[model_name] = {}
    results[model_name]['precision'] = np.mean(res['precision'])
    results[model_name]['recall'] = np.mean(res['recall'])
    results[model_name]['f1'] = np.mean(res['f1'])

with open("results.json", "w") as f:
    json.dump(results, f)