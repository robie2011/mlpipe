import os
from typing import List, Dict, Tuple


def get_dir(join_to_root_path=[]):
    root = os.path.dirname(__file__)
    return str(os.path.join(root, *join_to_root_path))


def get_dict_values(d: Dict, *keys: str):
    return [d.get(k) for k in keys]
