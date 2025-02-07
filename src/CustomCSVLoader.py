# https://python.langchain.com/v0.1/docs/modules/data_connection/document_loaders/custom/

from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document

import pandas as pd

class CustomCSVLoader(BaseLoader):
    def __init__(self, file:str):
        self.file = file

    def load(self)->list[Document]:
        csv = pd.read_csv(self.file)
        rows = csv.apply(lambda x: " ".join(x), axis=1).to_list()
        return [Document(page_content=row) for row in rows]