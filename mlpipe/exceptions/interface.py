class MLException(Exception):
    pass


class MLConfigurationError(MLException):
    pass


class MLCreateInstanceException(MLException):
    pass


class MLConfigurationNotFound(MLException):
    pass
