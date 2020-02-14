from typing import List


def write_text_file(path: str, text: str):
    with open(path, "w") as f:
        f.write(text)


def read_text_file_lines(path: str) -> List[str]:
    with open(path, "r") as f:
        return f.readlines()
