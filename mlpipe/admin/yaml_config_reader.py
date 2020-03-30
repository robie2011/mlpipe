import sys

from mlpipe.utils.file_handlers import read_yaml

if __name__ == "__main__":
    path_yaml = sys.argv[1]
    key = sys.argv[2]
    config = read_yaml(path_yaml)
    for k in key.split("."):
        config = config[k]

    print(config)

