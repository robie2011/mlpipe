import pandas as pd

data = None
DEBUG = False


def load_empa_data():
    global data
    if data is None:
        data = pd.read_pickle('/Users/robert.rajakone/repos/2019_p9/code/helpers/empa_data_201807_201907.pkl')

    if DEBUG:
        return data.iloc[:10000, :]
    return data
