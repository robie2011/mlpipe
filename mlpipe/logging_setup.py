import logging
import logging.config
import os

import coloredlogs
import yaml

from mlpipe.utils.dictionary_parser import DictionaryParser
from mlpipe.utils.path_tool import dir_code


def setup_logging(default_path='logging.yaml', default_level=logging.INFO, env_key='LOG_CFG'):
    """
    | **@author:** Prathyush SP
    | Logging Setup
    https://gist.github.com/kingspp/9451566a5555fb022215ca2b7b802f19
    """
    path = default_path
    value = os.getenv(env_key, None)
    if value:
        path = value

    path = dir_code / path
    if os.path.exists(path):
        with open(path, 'rt') as f:
            try:
                config = yaml.safe_load(f.read())
                logging.config.dictConfig(config)
                dict_parser = DictionaryParser(config, separator='/')
                fmt = dict_parser.get('coloredlogs/format', None)
                _level = dict_parser.get('coloredlogs/level', 'INFO')
                # noinspection PyProtectedMember
                level = logging._nameToLevel[_level]

                coloredlogs.install(fmt=fmt, level=level)
            except Exception as e:
                print(e)
                print('Error in Logging Configuration. Using default configs')
                logging.basicConfig(level=default_level)
                coloredlogs.install(level=default_level)
    else:
        logging.basicConfig(level=default_level)
        coloredlogs.install(level=default_level)
        print('Failed to load configuration file. Using default configs')
        print('given config file path: ', path)
