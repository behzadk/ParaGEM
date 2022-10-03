source ~/.bashrc
eval "$(conda shell.bash hook)"
conda create -n "paragem" python=3.8.0 ipython

conda activate paragem

# conda install -y scipy 
# conda install -y cobra
# conda install -y numba
# conda install -y matplotlib
# conda install -y cobra


pip install scipy
pip install cobra
pip install matplotlib
pip install plotly
pip install carveme
pip install datapane

pip install gurobipy
pip install pandas
pip install psutil

pip install sphinx
pip install webcolors

pip install hydra-core
pip install loguru

pip install smetana
pip install multiprocess

pip install cometspy

pip install -r requirements.txt