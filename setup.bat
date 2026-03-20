python -m venv venv
call venv\Scripts\activate

pip install -r requirements.txt
pip install --no-build-isolation git+https://github.com/graphdeco-inria/diff-gaussian-rasterization.git