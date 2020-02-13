from abc import ABC, abstractmethod


class AbstractModelLoader(ABC):
    @abstractmethod
    def load(self, obj: object):
        pass
