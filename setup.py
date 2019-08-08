#!/usr/bin/env python

from setuptools import setup, find_packages

setup(name='human_aware_rl',
      version='0.0.1',
      description='This package has shared components.',
      author='Micah Carroll',
      author_email='micah.d.carroll@berkeley.edu',
      packages=find_packages(),
      install_requires=[
        'GitPython',
        'memory_profiler',
        'sacred==0.7.4',
        'pymongo',
        'numpy==1.15.1',
        'matplotlib==3.0.3',
        'seaborn==0.9.0',
        'pygame==1.9.5'
      ],
    )