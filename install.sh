source ~/.bashrc
eval "$(conda shell.bash hook)"
conda create -n "cobrapy_env" python=3.8.0 ipython

conda activate cobrapy_env
conda install -y scipy 
conda install -y cobra
conda install -y numba
conda install -y matplotlib
conda install -y cobra

pip install loguru
pip install smetana
pip install cobra
