import os

training = "/tmp/mlpipe/training",
data_package = "/tmp/mlpipe/packages",
tmp = "/tmp/mlpipe/tmp"

for c in [training, data_package, tmp]:
    if not os.path.isdir(c):
        os.mkdir(c)
