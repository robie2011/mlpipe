from encoders.one_hot_encoder import OneHotEncoder


class RangeEncoder(OneHotEncoder):
    def __init__(self, value_from: int, value_to: int):
        super().__init__(encoding=range(value_from, value_to))
