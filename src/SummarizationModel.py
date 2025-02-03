from src.Model import Model
from transformers import (AutoModelForSeq2SeqLM, AutoTokenizer, SummarizationPipeline, pipeline)

class SummarizationModel(Model):
    def __init__(self, model:str, max_length:int=130, min_length:int=30, do_sample:bool=False):
        """
        Summarization model
        @param model: str - model name from Hugging Face
        """
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model)
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.agent:SummarizationPipeline = pipeline("summarization", model=self.model, tokenizer=self.tokenizer)
        self.max_length = max_length
        self.min_length = min_length
        self.do_sample = do_sample

    def invoke(self:str, context:str) -> str:
        """
        Invoke the model
        @param context: str - context
        @return str - summary
        """
        return self.agent(context, max_length=self.max_length, min_length=self.min_length, do_sample=self.do_sample)[0]['summary_text']