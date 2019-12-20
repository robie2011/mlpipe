from dataclasses import dataclass


@dataclass
class AppConfig:
    dir_training: str
    dir_data_package: str
    dir_tmp: str


def get_config():
    return AppConfig(
        dir_training="/tmp/mlpipe/training",
        dir_data_package="/tmp/mlpipe/packages",
        dir_tmp="/tmp/mlpipe/tmp"
    )

