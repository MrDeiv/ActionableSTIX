from langchain_ollama import ChatOllama
from langchain_ollama import OllamaEmbeddings
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser

class ChatModel:
    def __init__(self, model_name:str, retriever):
        self.model = ChatOllama(
            model=model_name
        )

        self.prompt = PromptTemplate.from_template(
            """
            <s> [INST]
            You are a cybersecurity assistant. You have been asked to answer the following question based on the given context.
            Your ansers must be technical and detailed. No introduction is required.
            [/INST] </s>
            [INST] Question: {question}
            Context: {context}
            Answer: [/INST]
            """
        )

        self.chain = ({"context": retriever, "question": RunnablePassthrough()}
                      | self.prompt
                      | self.model
                      | StrOutputParser())
        
    @property
    def model(self):
        return self._model
    
    @model.setter
    def model(self, value):
        self._model = value

    @property
    def prompt(self):
        return self._prompt
    
    @prompt.setter
    def prompt(self, value):
        self._prompt = value

    @property
    def chain(self):
        return self._chain
    
    @chain.setter
    def chain(self, value):
        self._chain = value

    def invoke(self, question:str):
        return self.chain.invoke(question)