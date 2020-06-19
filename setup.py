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
        'sacred',
        'pymongo',
        'numpy',
        'matplotlib',
        'seaborn==0.9.0',
        'pygame==1.9.5',
        'ray[rllib]'
      ],
    )