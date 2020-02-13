import os


def get_dir_from_code_root(join_to_root=()):
    root = os.path.dirname(__file__)
    root = os.path.dirname(root)
    root = os.path.dirname(root)

    return str(os.path.join(root, *join_to_root))


def get_dir(join_to_root_path=()):
    root = os.path.dirname(__file__)
    root = os.path.dirname(root)
    root = os.path.abspath(root)
    return str(os.path.join(root, *join_to_root_path))


