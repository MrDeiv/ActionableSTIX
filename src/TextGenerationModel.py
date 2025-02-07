from src.Model import Model
from transformers import (AutoModelForCausalLM, AutoTokenizer, TextGenerationPipeline, pipeline)
from sentence_transformers import SentenceTransformer
import numpy as np
from src.SummarizationModel import SummarizationModel
from typing import Tuple

class TextGenerationModel(Model):
    def __init__(self, model:str, max_new_tokens:int=512):
        """
        Text Generation model
        @param model: str - model name from Hugging Face
        """
        self.model = AutoModelForCausalLM.from_pretrained(model, trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
        self.agent:TextGenerationPipeline = pipeline(
            "text-generation", 
            model=self.model, 
            tokenizer=self.tokenizer, 
            trust_remote_code=True,
            do_sample=True)
        self.max_new_tokens = max_new_tokens
        self.summarization_model = SummarizationModel(
            model="facebook/bart-large-cnn"
        )

    def _compute_similarity(self, texts:list[str]) -> float:
        """
        Compute similarity between the generated texts
        @param embedds: list[str] - list of texts
        @return float - similarity
        """
        st = SentenceTransformer('all-MiniLM-L6-v2')
        embedds = [st.encode(a) for a in texts]
        similarities = []
        for i in range(3):
            for j in range(i+1, 3):
                similarities.append(st.similarity(embedds[i], embedds[j]))
        
        return np.mean(similarities)

    def invoke(self: str, question: str, prompt: str) -> Tuple[str, float]:
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

        answers = []
        temps = [0.1, 0.5, 1] 
        for t in temps:
            response = self.agent(messages, max_new_tokens=self.max_new_tokens, temperature=t)
            answers.append(response[0]['generated_text'][len(response[0]['generated_text'])-1]['content'])
        
        mean_similarity = self._compute_similarity(answers)
        
        if mean_similarity > 0.7:
            context = "\n".join(answers)
            summary = self.summarization_model.invoke(context)
            print(summary)
            return summary, mean_similarity
        else:
            return "Gathered information is not coherent", mean_similarity