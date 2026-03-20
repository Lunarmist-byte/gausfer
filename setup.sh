#!/bin/bash

set -e
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install --no-build-isolation git+https://github.com/graphdeco-inria/diff-gaussian-rasterization.git

echo "Setup complete!"