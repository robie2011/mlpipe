from dataclasses import dataclass
from pathlib import Path
from typing import List


@dataclass
class CsvStreamWriter:
    headers: List[str]
    path: Path
    seperator: str = ","

    def __enter__(self):
        is_new = not self.path.is_file()
        self.file = open(str(self.path), "a")
        if is_new:
            self.write(self.headers)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.file.flush()
        self.file.close()

    def write(self, values: List[str]):
        self.file.write(
            self.seperator.join(
                map(lambda v: str(v), values)) + "\r\n")
        self.file.flush()
