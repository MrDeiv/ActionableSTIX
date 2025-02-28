from langchain.output_parsers import ListOutputParser
import re

class ListParser(ListOutputParser):
    def parse(self, text: str) -> list[str]:
        reg = re.compile(r'[0-9]\.')
        first = reg.search(text)
        if first:
            text = text[first.start():]
        
        lines = text.split('\n')
        
        list_elements = []
        for line in lines:
            if reg.search(line):
                line_without_number = reg.sub('', line)
                list_elements.append(line_without_number.strip())

        return list_elements