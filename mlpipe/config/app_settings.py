from mlpipe.config.app_config_parser import AppConfigParser
from mlpipe.utils.file_tool import File
from mlpipe.utils.path_tool import dir_code


def get_config() -> AppConfigParser:
    config_dict = File.read_yaml(dir_code / 'app_config.yml')
    return AppConfigParser(config_dict)


AppConfig = get_config()
