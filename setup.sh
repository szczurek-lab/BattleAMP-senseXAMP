conda install pytorch==1.7.1 -c pytorch

pip install -r requirements.txt

python -c "import gdown; gdown.download_folder(url=\
'https://drive.google.com/drive/folders/1wNuoFrFZd3q3AlGyV-s2WpaVMs06N4L1',\
output='weights')"

mkdir -p datasets/ori_datasets
mkdir -p datasets/stc_datasets
mkdir -p datasets/esm_embeddings
mkdir -p datasets/stc_info
