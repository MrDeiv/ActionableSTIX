from langchain.prompts import PromptTemplate
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
from langchain_huggingface import HuggingFacePipeline, ChatHuggingFace

MODEL = "microsoft/Phi-3.5-mini-instruct"

if __name__ == "__main__":

    text = """
    Marie Curie, born in 1867, was a Polish and naturalised-French physicist and chemist who conducted pioneering research on radioactivity.
    She was the first woman to win a Nobel Prize, the first person to win a Nobel Prize twice, and the only person to win a Nobel Prize in two scientific fields.
    Her husband, Pierre Curie, was a co-winner of her first Nobel Prize, making them the first-ever married couple to win the Nobel Prize and launching the Curie family legacy of five Nobel Prizes.
    She was, in 1906, the first woman to become a professor at the University of Paris.
    """ 

    prompt = """
        Extract entities and their relationships from the following text in JSON format.

        Text:
        {text}
        """.format(text=text)

    model = AutoModelForCausalLM.from_pretrained(MODEL)
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, device="cpu")

    hfp = HuggingFacePipeline(pipeline=pipe, verbose=True)
    chat = ChatHuggingFace(llm=hfp)
    
    response = chat.invoke(prompt)
    print(response)
    