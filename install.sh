#!/bin/sh
cd baselines
python setup.py develop
cd ..

cd stable-baselines
python setup.py develop
cd ..

cd overcooked_ai
python setup.py develop
cd ..

cd tfjs-converter
yarn
cd ..

python setup.py develop