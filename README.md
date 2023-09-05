# FGNN: A scoring function to predict the binding affinity of protein-ligand complexes 

FGNN is a novel deep fusion graph neural networks framework named FGNN to learn the protein–ligand interactions from the 3D structures of protein–ligand complexes.
More information is published in the paper.(https://pubs.rsc.org/en/content/articlelanding/2023/cp/d3cp03651k)

## Usage of FGNN
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
