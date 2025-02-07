from tqdm import tqdm
from transformers import (AutoModelForSeq2SeqLM, AutoTokenizer, SummarizationPipeline, pipeline)
import evaluate
import datasets
from transformers import DataCollatorForSeq2Seq
import json

#model_name = 'google-t5/t5-small'
#model_name = 'facebook/bart-large-cnn'
ds = datasets.load_dataset("FiscalNote/billsum", split="ca_test")

rouge = evaluate.load('rouge')

scores = {}
models = [
    'Falconsai/text_summarization',
    'google-t5/t5-small',
    'google/pegasus-xsum',
    'T-Systems-onsite/mt5-small-sum-de-en-v2',
    'csebuetnlp/mT5_multilingual_XLSum'
]

progress = tqdm(total=len(models), desc="Evaluating")
for model_name in models:
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        agent = SummarizationPipeline(model=model, tokenizer=tokenizer)
        scores[model_name] = {}
        example = ds[0]
        source = example['text']
        target = example['summary']
        output = agent(source, max_length=512, min_length=30, do_sample=False)
        output = output[0]['summary_text']
        scores[model_name] = rouge.compute(predictions=[output], references=[target])
    except Exception as e:
        print(f"error: {e}")
        scores[model_name] = None

    progress.update(1)

progress.close()

with open('summarisation_eval.json', 'w') as f:
    json.dump(scores, f)