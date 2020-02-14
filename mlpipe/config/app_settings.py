import os
import yaml
from typing_extensions import TypedDict

from mlpipe.utils.path_tool import get_dir_from_code_root

train_enable_datasource_caching = True
dir_training = "/tmp/mlpipe/training"
dir_data_package = "/tmp/mlpipe/packages"
dir_tmp = "/tmp/mlpipe/tmp"
dir_analytics = "/tmp/mlpipe/analytics"
api_port = 5000
TEST_STANDARD_FORMAT_DISALBE_TIMESTAMP_CHECK = False

training_monitor = 'val_loss'

for c in [dir_training, dir_data_package, dir_tmp, dir_analytics]:
    if not os.path.isdir(c):
        os.makedirs(c, exist_ok=True)


def _get_config():
    path = get_dir_from_code_root(['app_config.yml'])
    with open(path) as f:
        return yaml.load(f, yaml.FullLoader)


class ReportingConfig(TypedDict):
    hosting_url_prefix: str
    base_href: str
    html_template_path: str
    output_path: str


def get_reporting_config() -> ReportingConfig:
    cfg = _get_config()
    return cfg['reporting']
