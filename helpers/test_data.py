import pandas as pd
import numpy as np
import os
from pathlib import Path

data = None
DEBUG = False


def load_empa_data() -> pd.DataFrame:
    global data
    if data is None:
        data_file = os.path.join(Path(os.path.realpath(__file__)).parent, 'empa_data_201807_201907.pkl')
        data = pd.read_pickle(data_file)

    if DEBUG:
        return data.iloc[:10000, :]
    return data


def load_simulation():
    data_file = os.path.join(Path(os.path.realpath(__file__)).parent, 'test_data.npy')
    return np.load(data_file, allow_pickle=True)
