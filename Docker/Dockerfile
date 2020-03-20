FROM robie2011/mlpipe-base
ADD ./ /opt/mlpipe/
RUN /bin/bash -c "pip install diskcache"
RUN /bin/bash -c "ln -s /opt/mlpipe/mlpipe.cli.sh /usr/bin/mlpipe && mkdir /data"
WORKDIR /data
ENTRYPOINT [ "/opt/mlpipe/mlpipe.cli.sh"]