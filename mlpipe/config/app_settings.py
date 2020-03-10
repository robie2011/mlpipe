from mlpipe.config.app_config_parser import AppConfigParser
from mlpipe.utils.file_handlers import read_yaml
from mlpipe.utils.path_tool import dir_code


def get_config() -> AppConfigParser:
    config_dict = read_yaml(dir_code / 'app_config.yml')
    return AppConfigParser(config_dict)


AppConfig = get_config()
