git clone https://github.com/mlcommons/inference.git
cd inference

# Install Python dependencies
pip install -r requirements.txt

# Additional dependencies might include
pip install numpy
pip install tensorflow  # or pytorch, etc.

# Example command to run a benchmark
python3 run_local.sh <benchmark_name> <scenario>

