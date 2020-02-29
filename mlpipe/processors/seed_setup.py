from mlpipe.processors.interfaces import AbstractProcessor
from mlpipe.processors.standard_data_format import StandardDataFormat


class SeedSetup(AbstractProcessor):
    def __init__(self, numpy=None, tensorflow=None):
        import random
        self.np_seed = numpy or random.randint(0, 1000)
        self.tf_seed = tensorflow or random.randint(0, 1000)
        self.logger.info(f"using np_seed {self.np_seed}")
        self.logger.info(f"using tf_seed {self.tf_seed}")

    def _process2d(self, processor_input: StandardDataFormat) -> StandardDataFormat:
        # only import if instance is created
        import tensorflow
        import numpy

        # results are not always reproduciable
        # https://machinelearningmastery.com/reproducible-results-neural-networks-keras/
        # https://www.tensorflow.org/api_docs/python/tf/random/set_seed?version=stable
        self.logger.info("using numpy random seed={0}".format(self.np_seed))
        self.logger.info("using tensorflow random seed={0}".format(self.tf_seed))
        numpy.random.seed(self.np_seed)

        tensorflow.random.set_seed(self.tf_seed)
        return processor_input
