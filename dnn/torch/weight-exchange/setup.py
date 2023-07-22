#!/usr/bin/env/python
import os
from setuptools import setup

lib_folder = os.path.dirname(os.path.realpath(__file__))

with open(os.path.join(lib_folder, 'requirements.txt'), 'r') as f:
    install_requires = list(f.read().splitlines())

print(install_requires)

setup(name='wexchange',
      version='1.4',
      author='Jan Buethe',
      author_email='jbuethe@amazon.de',
      description='Weight-exchange library between Pytorch and Tensorflow',
      packages=['wexchange', 'wexchange.tf', 'wexchange.torch', 'wexchange.c_export'],
      install_requires=install_requires
      )
