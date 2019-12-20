import os

enable_datasource_caching = True
dir_training = "/tmp/mlpipe/training"
dir_data_package = "/tmp/mlpipe/packages"
dir_tmp = "/tmp/mlpipe/tmp"

for c in [dir_training, dir_data_package, dir_tmp]:
    if not os.path.isdir(c):
        os.mkdir(c)
