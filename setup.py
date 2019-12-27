from distutils.core import setup
from setuptools import find_packages

with open('requirements.txt') as f:
    requirements = f.readlines()

# https://www.geeksforgeeks.org/command-line-scripts-python-packaging/
setup(name='mlpipe',
      version='1.0',
      packages=find_packages(),
      entry_points = {
            'console_scripts': [
                  'mlpipe = cli.mlpipe:main'
            ]
      },
      install_requires = requirements,
      zip_safe=False)
