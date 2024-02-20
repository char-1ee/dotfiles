time wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ${HOME}/miniconda.sh
time bash ${HOME}/miniconda.sh -b -p ${HOME}/miniconda3
source ${HOME}/miniconda3/etc/profile.d/conda.sh
conda init
conda config --set auto_activate_base false

# Set up conda envs
mkdir -p ${HOME}/pyenv
time conda create -p ${HOME}/pyenv/baseline python=3.8 -y
export PYTHONPATH=${HOME}/pyenv/baseline/lib/python3.8/site-packages
