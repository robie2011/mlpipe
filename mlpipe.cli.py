import logging

if __name__ == "__main__":
    # create a logging format
    #package_name = __name__.split(".")[0]
    package_logger = logging.getLogger("mlpipe")
    package_logger.setLevel(logging.DEBUG)

    # add the handlers to the logger
    formatter = logging.Formatter('%(asctime)s [%(name)s] (%(levelname)s) %(message)s')
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    package_logger.addHandler(ch)

    from mlpipe.cli import mlpipe
    mlpipe.main()