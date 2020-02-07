import logging

from mlpipe.workflows.utils import get_class_name


class InstanceLoggerMixin:
    _logger = None

    def get_logger(self):
        if not self._logger:
            self._logger = logging.getLogger(get_class_name(self))
        return self._logger