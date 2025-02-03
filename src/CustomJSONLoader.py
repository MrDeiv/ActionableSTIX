# https://python.langchain.com/v0.1/docs/modules/data_connection/document_loaders/custom/

from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document
import yaml
import json

class CustomJSONLoader(BaseLoader):
    def __init__(self, file:str):
        self.file = file

    def load(self):
        hnd = open(self.file, "r")
        data = json.loads(hnd.read())
        return data