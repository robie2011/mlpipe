from dataclasses import dataclass
from typing import List, Dict


@dataclass
class LabelSelection:
    indexes: List[int]
    names: List[str]
    indexes_unselected: List[int]
    names_unselected: List[str]


@dataclass
class LabelSelector:
    elements: List[str]

    def select(self, selection: List[str]) -> LabelSelection:
        ix_selection = []
        for name in selection:
            try:
                i = self.elements.index(name)
                ix_selection.append(i)
            except ValueError as e:
                raise ValueError(f"Can not find label '{name}'. Available labels are: {selection}")

        unselected = [(ix, name) for ix, name
                         in enumerate(self.elements)
                         if ix not in ix_selection]
        names_unselected = [name for ix, name in unselected]
        ix_unselected = [ix for ix, name in unselected]

        return LabelSelection(
            indexes=ix_selection,
            names=selection,
            indexes_unselected=ix_unselected,
            names_unselected=names_unselected)

    def without(self, labels: List[str]) -> LabelSelection:
        selected = [name for name in self.elements
                    if name not in labels]
        return self.select(selected)


@dataclass
class DictionaryParser:
    data: Dict
    separator: str

    def get(self, nested_key: str, default):
        _d = self.data
        for k in nested_key.split(self.separator):
            if k in _d:
                _d = _d[k]
            else:
                return default
        return _d
