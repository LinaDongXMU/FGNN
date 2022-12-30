# FGNN: Reliable prediction of the binding affinity of protein-ligand complexes with fusion of graph neural networks and 3D structure-based complex graph

The work has been submitted.

After download FGNN, you need to do these firstly:

mkdir data/cache

mkdir data/data_cache

mkdir pdbbind2016/testset


## 1. Environment
conda env create -f environment-data.yml

conda env create -f environment-model.yml

## 2. Data preprocessing
conda activate data

python preprocess_pdbbind.py

## 3. Traing models
conda activate model

python train.py

## 4. Test and predict
conda activate model

python predict.py
