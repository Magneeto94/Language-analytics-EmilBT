#!/usr/bin/env bash

VENVNAME=venv_ass6

python3 -m venv $VENVNAME
source $VENVNAME/bin/activate
pip install --upgrade pip

pip install ipython
pip install pandas
pip install numpy
pip install gensim
pip install sklearn
pip install tensorflow
pip install matplotlib
pip install pydot
pip install seaborn

test -f requirements.txt && pip install -r requirements.txt

deactivate