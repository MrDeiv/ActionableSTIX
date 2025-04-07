from langchain.output_parsers import ListOutputParser
import re

class ListParser(ListOutputParser):
    def parse(self, text: str) -> list[str]:
        reg_ol = re.compile(r'[0-9]\.')
        reg_ul = re.compile(r'\*.')

        first = reg_ol.search(text) or reg_ul.search(text)
        if first:
            text = text[first.start():]
        
        lines = text.split('\n')
        
        list_elements = []
        reg_symbol = re.compile(r'^\*.')
        reg_number = re.compile(r'^\d\.')

        for line in lines:
            if reg_number.search(line) or reg_symbol.search(line):
                if line.startswith('.'):
                    line = line[1:]
                line_without_number = reg_number.sub('', line)
                list_elements.append(line_without_number.strip())

        return list_elements