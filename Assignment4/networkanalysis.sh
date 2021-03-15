#!/usr/bin/env bash

VENVNAME=networkanalysis

python3 -m venv $VENVNAME
source $VENVNAME/bin/activate
pip install --upgrade pip

pip install ipython
pip install jupyter

python -m ipykernel install --user --name=$VENVNAME

test -f requirements.txt && pip install -r requirements.txt

pip install -U pip setuptools wheel
pip install -U spacy
python -m spacy download en_core_web_sm

deactivate
echo "build $VENVNAME"
