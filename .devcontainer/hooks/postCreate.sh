#!/bin/sh
poetry config virtualenvs.in-project true
poetry config virtualenvs.prompt venv
poetry install

sudo chown -R vscode:vscode /home/vscode/.cache/pypoetry
