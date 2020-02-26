FROM tensorflow/tensorflow:2.0.0b1-gpu-py3-jupyter
ADD requirements.txt /tmp/requirements.txt
ADD install_environment.sh /tmp/install_environment.sh
RUN cd /tmp/; bash /tmp/install_environment.sh
