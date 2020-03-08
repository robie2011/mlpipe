def write_json(path: str, data: object):
    import json
    with open(path, "w") as f:
        return json.dump(data, f, indent=4)


def read_json(path: str):
    import json
    with open(path, "r") as f:
        return json.load(f)


def read_text(path: str):
    with open(path, "r") as f:
        return f.read()


def write_binary(path: str, data: object):
    import pickle
    with open(path, "wb") as f:
        return pickle.dump(data, f)


def read_binary(path: str):
    import pickle
    with open(path, "rb") as f:
        return pickle.load(f)
