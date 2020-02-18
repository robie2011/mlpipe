class MLException(Exception):
    pass


class MLDslConfigurationException(MLException):
    pass


class MLPipeError(MLException):
    pass


class MLMissingConfigurationException(MLException):
    pass
