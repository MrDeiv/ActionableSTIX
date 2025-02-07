from src.Model import Model
from transformers import (AutoModelForCausalLM, AutoTokenizer, TextGenerationPipeline, pipeline)

class SentimentAnalysisModel(Model):
    def __init__(self, model:str, max_new_tokens:int=128):
        """
        Sentiment Analysis model
        @param model: str - model name from Hugging Face
        """
        self.model = AutoModelForCausalLM.from_pretrained(model, trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
        self.agent:TextGenerationPipeline = pipeline("text-generation", model=self.model, tokenizer=self.tokenizer, trust_remote_code=True)
        self.max_new_tokens = max_new_tokens

    def invoke(self:str, text:str) -> str:
        """
        Invoke the model
        @param text: str - text
        @return str - generated text
        """
        prompt = f"""
        You must provide the sentiment of the statement.
        The sentiment must be one of the following:
        - True (for positive sentiment)
        - False (for negative sentiment)
        """
        
        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": text},
        ]

        response = self.agent(messages, max_new_tokens=self.max_new_tokens)
        response = response[0]['generated_text'][len(response[0]['generated_text'])-1]['content']

        try:
            response = bool(response.strip())
        except Exception as e:
            response = 'Unknown'
            print("Error: ", e)

        return response