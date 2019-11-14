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
            print(f"group={row} ", xxs[row, :, sensor])
        print("")


def generated_3d_data(size=(3, 5, 4)):
    np.random.seed(1)
    data = np.round(np.random.random(size) * 1000)
    return data, indexes


# class DataBuilder3D:
#     axis0: int = 0
#     axis1: int = 0
#     axis2: int = 0
#
#     def __init__(self, axis0, axis1, axis2):
#         self.data = np.nan((axis0, axis1, axis2))
#
#     def set_z_axis(self, axis0: int):
#         self.axis0 = axis0


def transform_to_2d_matrix(data: np.ndarray):
    if len(data.shape) > 1:
        print("NOTE: Data already in 2D shape")

    return data if len(data.shape) > 1 else data.reshape(-1, 1)
