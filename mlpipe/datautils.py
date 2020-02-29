import re
from dataclasses import dataclass
from typing import List

from mlpipe.mixins.logger_mixin import InstanceLoggerMixin


@dataclass
class LabelSelection:
    indexes: List[int]
    names: List[str]
    indexes_unselected: List[int]
    names_unselected: List[str]


@dataclass
class LabelSelector(InstanceLoggerMixin):
    elements: List[str]

    def select(self, selection: List[str], enable_regex=False) -> LabelSelection:
        logger = self.logger
        ix_selection = []
        for name in selection:
            if enable_regex and name.startswith("REGEX:"):
                pattern = name.split("REGEX:")[-1]
                logger.debug(f"looking for regex: {pattern} in list {self.elements}")
                regex_match = [ix for ix, real_name in enumerate(self.elements) if re.search(pattern, real_name)]
                ix_selection += regex_match
                logger.debug(f"found: {[self.elements[ix] for ix in regex_match]}")
                if len(regex_match) == 0:
                    raise ValueError(f"Regex pattern has zero match: {pattern}")
            else:
                try:
                    i = self.elements.index(name)
                    ix_selection.append(i)
                except ValueError as e:
                    raise ValueError(
                        f"Can not find label '{name}'. Available labels are: {self.elements}. Error: {e.args}")

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
