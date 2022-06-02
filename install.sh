#!/bin/sh

cp setup_corrections/setup_baselines.py baselines/setup.py
cp setup_corrections/setup_stable_baselines.py stable-baselines/setup.py
cp setup_corrections/setup_main.py setup.py
cp setup_corrections/setup_overcooked.py overcooked_ai/setup.py

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
