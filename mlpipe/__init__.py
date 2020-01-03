import logging

package_logger = logging.getLogger(__name__)
package_logger.setLevel(logging.DEBUG)

# add the handlers to the logger
formatter = logging.Formatter('%(asctime)s [%(name)s] (%(levelname)s) %(message)s')
ch = logging.StreamHandler()
ch.setFormatter(formatter)
package_logger.addHandler(ch)