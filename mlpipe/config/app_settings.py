import os

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
