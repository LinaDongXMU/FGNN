mkdir results
python prepare_files.py
python process/preprocess_pdbbind.py --data_path_core ..data/coreset --data_path_refined ..data/refined-set --dataset_name pdbbind2016 --output_path results --cutoff 5.5
