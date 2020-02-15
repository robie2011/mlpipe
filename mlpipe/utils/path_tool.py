from pathlib import Path

dir_mlpipe = Path(__file__).parent.parent
dir_code = dir_mlpipe.parent


def get_abspath_or_relpath(path: str):
    if Path(path).is_absolute():
        return Path(path)
    else:
        return dir_code / path
