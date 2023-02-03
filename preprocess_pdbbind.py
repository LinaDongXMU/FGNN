"""
Preprocessing code for the protein-ligand complex.
"""
import argparse
import os

from utils import *
from process.featurizer import *


def process_dataset(protein_name, test_lst, path, cutoff):
    # atomic sets for long-range interactions
    atom_types = [6,7,8,9,15,16,17,35,53]
    atom_types_ = [6,7,8,16]
    # atomic feature generation
    featurizer = Featurizer(save_molecule_codes=False)
    processed_dict = {}
    processed_dict[protein_name] = gen_feature(path, protein_name, featurizer)
    processed_dict = pairwise_atomic_types(path, processed_dict, atom_types, atom_types_)

    # load pka (binding affinity) data
    pk_dict = load_pk_data(path+'index/INDEX_general_PL_data.2016')
    data_dict = processed_dict
    for k,v in processed_dict.items():
        v['pk'] = pk_dict[k]
        data_dict[k] = v

    training_id, training_data, training_pk = [], [], []
    test_id, test_data, test_pk = [], [], []
    fault=[]
    for k, v in data_dict.items():
        ligand = (v['lig_fea'], v['lig_co'], v['lig_atoms'], v['lig_eg'])
        pocket = (v['pock_fea'], v['pock_co'], v['pock_atoms'], v['pock_eg'])
        try:
            graph = cons_lig_pock_graph_with_spatial_context(ligand, pocket, add_fea=3, theta=cutoff, keep_pock=False, pocket_spatial=True)
            cofeat, pk = v['type_pair'], v['pk']
            graph = list(graph) + [cofeat]
            if k in test_lst:
                test_id.append(k)
                test_data.append(graph)
                test_pk.append(pk)
                continue
            training_id.append(k)
            training_data.append(graph)
            training_pk.append(pk)
        except:
            fault.append(k)
    # split train and valid
    train_idxs, valid_idxs = random_split(len(training_data), split_ratio=1, seed=2020, shuffle=True)
    train_i = [training_id[i] for i in train_idxs]
    train_g = [training_data[i] for i in train_idxs]
    train_y = [training_pk[i] for i in train_idxs]
    valid_i = [training_id[i] for i in valid_idxs]
    valid_g = [training_data[i] for i in valid_idxs]
    valid_y = [training_pk[i] for i in valid_idxs]
    train = (train_i, train_g, train_y)
    valid = (valid_i, valid_g, valid_y)
    test = (test_id, test_data, test_pk)
    return train, valid, test

def pocket_mol2(path):
     for filename in os.listdir(path):
         pocket_pdb=os.path.join(path,filename,str(filename)+'_pocket.pdb')
         pocket_mol2=os.path.join(path,filename,str(filename)+'_pocket.mol2')
         cmd='obabel -ipdb '+str(pocket_pdb)+' -omol2 -O '+str(pocket_mol2)
         os.system(cmd)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path_test', type=str, default='./pdbbind2016/testset')
    parser.add_argument('--data_path_training', type=str, default='./pdbbind2016/trainingset')
    parser.add_argument('--output_path', type=str, default='./data/')
    parser.add_argument('--dataset_name', type=str, default='pdbbind2016')
    parser.add_argument('--n_jobs', type=int, default=-1)
    parser.add_argument('--cutoff', type=float, default=5.5)
    parser.add_argument('--file_process', type=bool, default=True)
    args = parser.parse_args()

    if args.file_process:
        print('processing file')
        pocket_mol2(args.data_path_test)
        pocket_mol2(args.data_path_training)
        print('done')
    else:
        print('ignore file process')

    test_set_list = [x for x in os.listdir(args.data_path_test)]
    training_set_list = [x for x in os.listdir(args.data_path_training)]

    print('processing dataset')
    data = pmap_multi(process_dataset, zip(training_set_list),
                      n_jobs=args.n_jobs,
                      desc='Get receptors',
                      test_lst=test_set_list,
                      path=args.data_path_training,
                      cutoff=args.cutoff)
    write_pickle(data, args.output_path, args.dataset_name)
    print('done')