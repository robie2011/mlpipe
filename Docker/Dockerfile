FROM python:3.6
# python 3.7 is not yet compatible with PyYAML
# python 3.8 is not yet compatible with tensorflow
ADD *:wq
RUN pip install dataclass && python setup.py install
