#

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
