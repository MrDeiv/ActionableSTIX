import os
import json

class VTParser:
    @staticmethod
    def parse_json(file:str) -> str:
        assert os.path.exists(file), f"File {file} not found"

        # load json
        json_content = json.loads(open(file).read())
        attributes = json_content['Event']['Attribute']

        out = []
        for attribute in attributes:
            out.append(attribute['type'] + " " + attribute['value'])

        return "\n".join(out)