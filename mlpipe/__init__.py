from mlpipe.logging_setup import setup_logging

# package_logger = logging.getLogger(__name__)
#package_logger.setLevel(logging.DEBUG)

# logging coloring library
# https://pypi.org/project/coloredlogs/#usage
# coloredlogs.install(logger=package_logger, fmt=LOGGING_FORMAT)


# builtin logging handler
# formatter = logging.Formatter(LOGGING_FORMAT)
# ch = logging.StreamHandler()
# ch.setFormatter(formatter)
# package_logger.addHandler(ch)


setup_logging()