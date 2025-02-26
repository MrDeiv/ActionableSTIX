from langchain.output_parsers import ListOutputParser
import re

class ListParser(ListOutputParser):
    def parse(self, text: str) -> list[str]:
        reg = re.compile(r'[0-9]\.')
        first = reg.search(text)
        if first:
            text = text[first.start():]
        
        return text.split('\n')