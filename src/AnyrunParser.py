import os

class AnyrunParser:
    @staticmethod
    def parse_txt(file:str) -> str:
        assert os.path.exists(file), f"File {file} not found"
        lines = open(file).readlines()

        out = []
        prefix = ""
        for line in lines:
            if line.startswith(" "):
                out.append(prefix + line.strip())
            else:
                prefix = line.strip() + " = "
        return "\n".join(out)