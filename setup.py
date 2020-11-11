#!/usr/bin/env python

from setuptools import setup, find_packages

setup(name='human_aware_rl',
      version='0.0.1',
      description='This package has shared components.',
      author='Micah Carroll',
      author_email='micah.d.carroll@berkeley.edu',
      packages=find_packages(),
      install_requires=[
        'opencv-python==4.2.0.32',
        'opencv-python-headless==4.2.0.32',
        'GitPython',
        'memory_profiler',
        'sacred',
        'pymongo',
        'numpy',
        'dill',
        'matplotlib',
        'seaborn==0.9.0',
        'pygame==1.9.5',
        'ray[rllib]==0.8.5'
      ],
    )
