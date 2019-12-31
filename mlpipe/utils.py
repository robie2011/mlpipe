import os


def get_dir(join_to_root_path=[]):
    root = os.path.dirname(__file__)
    return str(os.path.join(root, *join_to_root_path))

