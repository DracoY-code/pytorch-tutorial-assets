#!/bin/zsh

# Get the current directory
current_dir=$(pwd)

# Append the `scripts` directory to PYTHONPATH
export PYTHONPATH=$PYTHONPATH:$current_dir/scripts

# Run the tests
pytest -s
