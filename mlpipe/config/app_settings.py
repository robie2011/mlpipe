import os

from mlpipe.config.app_config_parser import AppConfigParser
from mlpipe.utils.file_tool import File
from mlpipe.utils.path_tool import dir_code

dir_training = "/tmp/mlpipe/training"
dir_tmp = "/tmp/mlpipe/tmp"
TEST_STANDARD_FORMAT_DISALBE_TIMESTAMP_CHECK = False

training_monitor = 'val_loss'

for c in [dir_training, dir_tmp]:
    if not os.path.isdir(c):
        os.makedirs(c, exist_ok=True)


def get_config() -> AppConfigParser:
    config_dict = File.read_yaml(dir_code / 'app_config.yml')
    return AppConfigParser(config_dict)


AppConfig = get_config()

