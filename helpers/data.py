import yaml
import numpy as np


def load_yaml(file_path: str):
    with open(file_path) as f:
        content = f.read()
        return yaml.load(content)


def print_3d_array(xxs: np.ndarray):
    rows, window_size, sensors = xxs.shape
    for sensor in range(sensors):
        print("sensor ", sensor)
        for row in range(rows):
            print(f"t={row} ", data[row, :, sensor])
        print("")


def generated_3d_data(size=(3, 5, 4)):
    np.random.seed(1)
    return np.round(np.random.random(size) * 1000)
