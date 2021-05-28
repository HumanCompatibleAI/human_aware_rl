#!/bin/sh
cd overcooked_ai
pip install -e .
cd ..

pip install -e .

conda install protobuf -y