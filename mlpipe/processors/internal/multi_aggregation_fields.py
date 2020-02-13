from dataclasses import dataclass
from typing import List


@dataclass
class MissingFields(BaseException):
    fields: List[str]


@dataclass
class MissingFieldsForLogic(MissingFields):
    logic: object

    def __str__(self):
        return "For logic: \n\t{0} \n\tfollowing fields are missing as input variable: {1}".format(
            self.logic,
            ", ".join(self.fields)
        )
