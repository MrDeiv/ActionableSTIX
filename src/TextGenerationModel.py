from src.Model import Model
from transformers import (AutoModelForCausalLM, AutoTokenizer, TextGenerationPipeline, pipeline)
from src.SummarizationModel import SummarizationModel

class TextGenerationModel(Model):
    def __init__(self, model:str, max_new_tokens:int=1024, temperature:float=0.1):
        """
        Text Generation model
        @param model: str - model name from Hugging Face
        """
        self.model = AutoModelForCausalLM.from_pretrained(model, trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model, 
            trust_remote_code=True)
        #self.tokenizer.chat_template = "<prompt_template>"
        self.agent:TextGenerationPipeline = pipeline(
            "text-generation", 
            model=self.model, 
            tokenizer=self.tokenizer, 
            trust_remote_code=True,
            do_sample=True,
            device='cpu')
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

    def invoke(self: str, question: str, prompt: str) -> str:
        """
        Invoke the model
        @param question: str - question
        @param prompt: str - prompt
        @return str - generated text
        """
        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": question},
        ]

        response = self.agent(messages, max_new_tokens=self.max_new_tokens, temperature=self.temperature)
        return response[0]['generated_text'][len(response[0]['generated_text'])-1]['content']