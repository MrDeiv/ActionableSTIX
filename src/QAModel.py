from src.Model import Model
from transformers import (AutoModelForQuestionAnswering, AutoTokenizer, QuestionAnsweringPipeline, pipeline)

class QAModel(Model):
    def __init__(self, model:str):
        """
        Question Answering Model
        @param model: str - model name from Hugging Face
        """
        self.model = AutoModelForQuestionAnswering.from_pretrained(model)
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.agent:QuestionAnsweringPipeline = pipeline("question-answering", model=self.model, tokenizer=self.tokenizer)

    def invoke(self, question:str, context:str) -> str:
        """
        Invoke the model
        @param question: str - question
        @param context: str - context
        @return str - answer
        """
        return self.agent(question=question, context=context)['answer']