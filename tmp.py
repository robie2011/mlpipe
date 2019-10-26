from abc import ABC, abstractmethod
import numpy as np

class AbstractGrouper(ABC):
    @abstractmethod
    def calculate(self, row: np.ndarray) -> np.array:
        """returns an np array of int for group identification"""
        pass


class NestedGrouper():
    def __init__(self, groupers: [AbstractGrouper]):
        self.groupers = groupers
        self.data = {}

    def add(self, index: int, row: np.ndarray):
        parent_group = self.data

        for grouper in self.groupers:
            group_id = grouper.get_group_id(row)

            # ensure new parament object is a container
            # for subgroups
            if parent_group[group_id] is None:
               parent_group[group_id] = {}

            parent_group = parent_group[group_id]

        if parent_group.values is None:
           parent_group.values = []
           parent_group.metrics = []
        parent_group.values.add(index)

    def group(self, data: np.ndarray):
        # make numpy array immutable
        # https://www.w3resource.com/python-exercises/numpy/python-numpy-exercise-67.php
        # use parallel libs

        # each column contains group identification for
        group_matrix = np.fromiter(map(lambda grouper: grouper.calculate_groups(data), self.groupers), dtype='int')
        # todo: sort by first, second, ... , last column
        # todo: find out where group changes and split there into separate groups
        # then we can put these small groups into separate threads to calculate analytics
