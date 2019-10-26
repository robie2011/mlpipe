import unittest
from preprocessors.chain_processor import init_processors, ProcessInitializationFailedException
from preprocessors import NanRemover
from test.preprocessors.dummy_processors_args import ProcessorWithArgs
from helpers.data import load_yaml


class ChainProcessorTests(unittest.TestCase):
    def test_invalid_processor_argument(self):
        config = load_yaml('sample_notebook_invalid_arg.yml')
        config_processors = config['preprocessors']
        self.assertRaises(ProcessInitializationFailedException,
                          lambda: init_processors(config_processors))

    def test_valid_processor(self):
        config = load_yaml('sample_notebook.yml')
        config_processors = config['preprocessors'][:2]
        processors = init_processors(config_processors)
        self.assertEqual(len(processors), 2)
        self.assertIsInstance(processors[0], NanRemover)
        self.assertIsInstance(processors[1], ProcessorWithArgs)


if __name__ == '__main__':
    unittest.main()
