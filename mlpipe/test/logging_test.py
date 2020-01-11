import logging

print("module name", __name__)
module_logger = logging.getLogger(__name__)

module_logger.info("info")
module_logger.debug("debuug")

print("end")