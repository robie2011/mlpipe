from typing import List


def write_text_file(path: str, text: str):
    with open(path, "w") as f:
        f.write(text)


def read_text_file_lines(path: str) -> List[str]:
    with open(path, "r") as f:
        return f.readlines()


class File:
    @staticmethod
    def read_yaml(path):
        import yaml
        with open(path) as f:
            return yaml.load(f, yaml.FullLoader)
