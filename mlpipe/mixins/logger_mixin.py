import logging

from mlpipe.workflows.utils import get_qualified_name


class InstanceLoggerMixin:
    _logger = None

    def get_logger(self):
        if not self._logger:
            self._logger = logging.getLogger(get_qualified_name(self))
        return self._logger

