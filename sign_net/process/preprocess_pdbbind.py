"""
Preprocessing code for the protein-ligand complex.
"""
import argparse
import os
import pickle
from functools import partial

import numpy as np
import openbabel
from openbabel import pybel
from scipy.spatial import distance, distance_matrix

from featurizer import Featurizer
from utils import *


def pocket_atom_num_from_mol2(name, path): # 输入为pdbid和路径
    n = 0
    with open('%s/%s/%s_pocket.mol2' % (path, name, name)) as f: # 打开pocket.mol2文件
        for line in f:
            if '<TRIPOS>ATOM' in line: # 从ATOM
                break
        for line in f:
            cont = line.split()
            if '<TRIPOS>BOND' in line or cont[7] == 'HOH': # 到BOND，且不是水
                break
            n += int(cont[5][0] != 'H') # 数口袋中非氢原子的数量
    return n # 该函数返回口袋中非氢非水原子的数量

def pocket_atom_num_from_pdb(name, path): # 输入为pdbid和路径
    n = 0
    with open('%s/%s/%s_pocket.pdb' % (path, name, name)) as f: # 打开pocket.pdb文件
        for line in f:
            if 'REMARK' in line: # 从REMARK开始
                break
        for line in f:
            cont = line.split()
            # break
            if cont[0] == 'CONECT': # 到CONECT
                break
            n += int(cont[-1] != 'H' and cont[0] == 'ATOM') # 非氢原子的ATOM的数量
    return n # 该函数返回口袋中非氢非水原子的数量

## function -- feature
def gen_feature(path, name, featurizer): # 输入路径和pdbid，featurizer的module
    charge_idx = featurizer.FEATURE_NAMES.index('partialcharge')
    ligand = next(pybel.readfile('mol2', '%s/%s/%s_ligand.mol2' % (path, name, name))) # 用pybel读入配体的mol2
    ligand_coords, ligand_features = featurizer.get_features(ligand, molcode=1) # 调用featurizer的get_feature函数，返回配体的坐标和特征
    pocket = next(pybel.readfile('mol2' ,'%s/%s/%s_pocket.mol2' % (path, name, name))) # 用pybel读入口袋的mol2
    pocket_coords, pocket_features = featurizer.get_features(pocket, molcode=-1) # 返回口袋的坐标和特征
    node_num = pocket_atom_num_from_mol2(name, path) # 口袋得节点数量
    pocket_coords = pocket_coords[:node_num] # 口袋节点的坐标
    pocket_features = pocket_features[:node_num] # 口袋节点的特征
    try:
        assert (ligand_features[:, charge_idx] != 0).any() # 配体电荷的标签至少一个不为零
        assert (pocket_features[:, charge_idx] != 0).any() # 口袋
        assert (ligand_features[:, :9].sum(1) != 0).all() # 配体前九个特征求和为一，全部（原子种类的one-hot）
    except:
        print(name)
    lig_atoms, pock_atoms = [], []
    for i, atom in enumerate(ligand):
        if atom.atomicnum > 1: # atom.atomicnum得到的是？原子序数？？？
            lig_atoms.append(atom.atomicnum) # 配体原子列表的构建
    for i, atom in enumerate(pocket):
        if atom.atomicnum > 1:
            pock_atoms.append(atom.atomicnum) # 口袋原子列表的构建
    for x in pock_atoms[node_num:]:
        assert x == 8
    pock_atoms = pock_atoms[:node_num] # 口袋原子列表只取到口袋节点数
    assert len(lig_atoms)==len(ligand_features) and len(pock_atoms)==len(pocket_features) # 两个体系的原子数列表长度和特征列表长度相等
    
    ligand_edges = gen_pocket_graph(ligand) # 配体的邻接矩阵
    pocket_edges = gen_pocket_graph(pocket) # 蛋白的邻接矩阵
    # print(pocket_edges)
    return {'lig_co': ligand_coords, 'lig_fea': ligand_features, 'lig_atoms': lig_atoms, 'lig_eg': ligand_edges, 'pock_co': pocket_coords, 'pock_fea': pocket_features, 'pock_atoms': pock_atoms, 'pock_eg': pocket_edges}
    # 这个函数分别返回配体、口袋的坐标、特征、原子、邻接矩阵
## function -- pocket graph
def gen_pocket_graph(pocket): # 输入pybel形式的分子
    edge_l = []
    idx_map = [-1]*(len(pocket.atoms)+1)
    idx_new = 0
    for atom in pocket:
        edges = []
        a1_sym = atom.atomicnum
        a1 = atom.idx
        if a1_sym == 1:
            continue
        idx_map[a1] = idx_new
        idx_new += 1
        for natom in openbabel.OBAtomAtomIter(atom.OBAtom):
            if natom.GetAtomicNum() == 1:
                continue
            a2 = natom.GetIdx()
            bond = openbabel.OBAtom.GetBond(natom,atom.OBAtom)
            bond_t = bond.GetBondOrder()
            edges.append((a1,a2,bond_t))
        edge_l += edges
    edge_l_new = []
    for a1,a2,t in edge_l:
        a1_, a2_ = idx_map[a1], idx_map[a2]
        assert((a1_!=-1)&(a2_!=-1))
        edge_l_new.append((a1_,a2_,t))
    return edge_l_new # 返回邻接矩阵

def dist_filter(dist_matrix, theta): # 输入距离矩阵，和阈值
    pos = np.where(dist_matrix<=theta)
    ligand_list, pocket_list = pos
    return ligand_list, pocket_list # 返回阈值内的配体列表和蛋白列表

def pairwise_atomic_types(path, processed_dict, atom_types, atom_types_): # 原子对类别数，输入处理字典，和两个原子类型列表
    keys = [(i,j) for i in atom_types_ for j in atom_types] # i是配体的，j是蛋白的
    # for name in os.listdir(path):
    #     if len(name) != 4:
    #         continue
    name = list(processed_dict.keys())[0]
    ligand = next(pybel.readfile('mol2', '%s/%s/%s_ligand.mol2' % (path, name, name))) # 读配体的mol2，转成pybel的分子
    pocket = next(pybel.readfile('pdb' ,'%s/%s/%s_protein.pdb' % (path, name, name))) # 读蛋白的pdb，转成pybel的分子
    coords_lig = np.vstack([atom.coords for atom in ligand]) # 配体的坐标
    coords_poc = np.vstack([atom.coords for atom in pocket]) # 蛋白的坐标
    atom_map_lig = [atom.atomicnum for atom in ligand] # 配体原子序号列表
    atom_map_poc = [atom.atomicnum for atom in pocket] # 蛋白原子序号列表
    dm = distance_matrix(coords_lig, coords_poc) #配体和蛋白坐标形成的距离矩阵
    ligs, pocks = dist_filter(dm, 12) # 以12为阈值过滤距离矩阵
    
    fea_dict = {k: 0 for k in keys} # 特征字典，初始化的值为0
    for x, y in zip(ligs, pocks):
        x, y = atom_map_lig[x], atom_map_poc[y] # 分别返回原子x、y的原子序号
        if x not in atom_types or y not in atom_types_: continue # 不在对应列表不操作
        fea_dict[(y, x)] += 1 # 在的时候就计数，+1
    processed_dict[name]['type_pair'] = list(fea_dict.values()) # 处理字典以id和'typy_pair'为键的值赋予特征字典的值列表

    return processed_dict # 返回处理字典

def load_pk_data(data_path): # 输入路径
    res = dict()
    with open(data_path) as f:
        for line in f:
            if '#' in line:
                continue
            cont = line.strip().split()
            if len(cont) < 5:
                continue
            code, pk = cont[0], cont[3]
            res[code] = float(pk)
    return res # res为一个键为id，值为label的列表

def get_lig_atom_types(feat):
    pos = np.where(feat[:,:9]>0)
    src_list, dst_list = pos
    return dst_list
def get_pock_atom_types(feat):
    pos = np.where(feat[:,18:27]>0)
    src_list, dst_list = pos
    return dst_list

def cons_spatial_gragh(dist_matrix, theta=5): # 构建空间图，输入距离矩阵和阈值
    pos = np.where((dist_matrix<=theta)&(dist_matrix!=0))
    src_list, dst_list = pos
    dist_list = dist_matrix[pos]
    edges = [(x,y) for x,y in zip(src_list, dst_list)]
    return edges, dist_list # 返回邻接矩阵，距离列表

def cons_mol_graph(edges, feas): # 构建分子图，输入邻接矩阵和特征
    size = feas.shape[0]
    edges = [(x,y) for x,y,t in edges]
    return size, feas, edges # 返回大小，特征，邻接矩阵

def pocket_subgraph(node_map, edge_list, pock_dist): # 构建口袋子图
    edge_l = []
    dist_l = []
    node_l = set()
    for coord, dist in zip(edge_list, np.concatenate([pock_dist, pock_dist])):
        x,y = coord
        if x in node_map and y in node_map:
            x, y = node_map[x], node_map[y]
            edge_l.append((x,y))
            dist_l.append(dist)
            node_l.add(x)
            node_l.add(y)
    dist_l = np.array(dist_l)
    return edge_l, dist_l # 输出邻接矩阵和距离矩阵

def edge_ligand_pocket(dist_matrix, lig_size, theta=4, keep_pock=False, reset_idx=True): # 配体和口袋的邻接矩阵，输入是距离矩阵和配体大小
    
    pos = np.where(dist_matrix<=theta) # 距离小于阈值的位置
    ligand_list, pocket_list = pos
    if keep_pock: # False
        node_list = range(dist_matrix.shape[1])
    else:
        node_list = sorted(list(set(pocket_list))) # 节点列表是口袋列表去重排序的列表
    node_map = {node_list[i]:i+lig_size for i in range(len(node_list))} # 每个口袋节点和配体的每个原子进行索引 ################################口袋节点索引：复合图节点索引
    
    dist_list = dist_matrix[pos] # 距离列表由距离小于阈值的位置所定义
    if reset_idx: # True
        edge_list = [(x,node_map[y]) for x,y in zip(ligand_list, pocket_list)] # 邻接矩阵是元组形式的
    else:
        edge_list = [(x,y) for x,y in zip(ligand_list, pocket_list)]
    
    edge_list += [(y,x) for x,y in edge_list] # 邻接矩阵反过来的
    dist_list = np.concatenate([dist_list, dist_list]) # 距离列表为了和邻接矩阵长度相对应所以double了
    
    return dist_list, edge_list, node_map # 返回距离列表，邻接矩阵和节点地图

def add_identity_fea(lig_fea, pock_fea, comb=1): # 添加确认体系的特征，输入时配体特征和口袋特征
    if comb == 1: # 是这种情况
        lig_fea = np.hstack([lig_fea, [[1]]*len(lig_fea)]) # 配体是特征长度*1
        pock_fea = np.hstack([pock_fea, [[-1]]*len(pock_fea)]) # 口袋是特征长度*（-1）
    elif comb == 2:
        lig_fea = np.hstack([lig_fea, [[1,0]]*len(lig_fea)])
        pock_fea = np.hstack([pock_fea, [[0,1]]*len(pock_fea)])
    else:
        lig_fea = np.hstack([lig_fea, [[0]*lig_fea.shape[1]]*len(lig_fea)])
        if len(pock_fea) > 0:
            pock_fea = np.hstack([[[0]*pock_fea.shape[1]]*len(pock_fea), pock_fea])
    
    return lig_fea, pock_fea # 返回添加了标识符的配体特征，口袋特征

def get_neighbors(one_hot_src, dst_idx):
    select_idx = np.nonzero(one_hot_src)
    neighbors = dst_idx[select_idx]
    return neighbors.tolist()

def D3_info(a, b, c):
    # 空间夹角
    ab = b - a  # 向量ab
    ac = c - a  # 向量ac
    cosine_angle = np.dot(ab, ac) / (np.linalg.norm(ab) * np.linalg.norm(ac))
    cosine_angle = cosine_angle if cosine_angle >= -1.0 else -1.0
    angle = np.arccos(cosine_angle)
    # 三角形面积
    ab_ = np.sqrt(np.sum(ab ** 2))
    ac_ = np.sqrt(np.sum(ac ** 2))  # 欧式距离
    area = 0.5 * ab_ * ac_ * np.sin(angle)
    return np.degrees(angle), area, ac_

def D3_info_cal(nodes_ls, coords):
    if len(nodes_ls) > 2:
        Angles = []
        Areas = []
        Distances = []
        for node_id in nodes_ls[2:]:
            angle, area, distance = D3_info(coords[nodes_ls[0]], coords[nodes_ls[1]],
                                            coords[node_id])
            Angles.append(angle)
            Areas.append(area)
            Distances.append(distance)
        return [np.max(Angles) * 0.01, np.sum(Angles) * 0.01, np.mean(Angles) * 0.01, np.max(Areas), np.sum(Areas),
                np.mean(Areas),
                np.max(Distances) * 0.1, np.sum(Distances) * 0.1, np.mean(Distances) * 0.1]
    else:
        return [0, 0, 0, 0, 0, 0, 0, 0, 0]
    
def get_3d_feature(edge_index, coords):
    src_idx, dst_idx = edge_index

    neighbors_ls = []
    for i, src_node in enumerate(src_idx):
        src_node = src_node.tolist()
        dst_node = dst_idx[i].tolist()
        tmp = [src_node, dst_node]  # the source node id and destination id of an edge
        one_hot_src = (src_node == src_idx) #.long()
        neighbors = get_neighbors(one_hot_src, dst_idx)
        neighbors.remove(dst_node)
        tmp.extend(neighbors)   # [[8,3,4,7], [8,4,3,7],[8,7,3,4]]
        neighbors_ls.append(tmp)   # 保存当前节点和他的邻居节点[[a_1,N(a_1)], [a_2,N(a_2)], ..., [a_n,N(a_n)]]
    edge_feature = list(map(partial(D3_info_cal, coords=coords), neighbors_ls))
    return np.array(edge_feature)

def cons_lig_pock_graph_with_spatial_context(ligand, pocket, add_fea=2, theta=5, keep_pock=False, pocket_spatial=True): # 复合图函数，输入pybel形式的配体分子和口袋分子
    lig_fea, lig_coord, lig_atoms_raw, lig_edge = ligand # 特征，坐标，原子，邻接矩阵
    pock_fea, pock_coord, pock_atoms_raw, pock_edge = pocket
    
    # inter-relation between ligand and pocket
    lig_size = lig_fea.shape[0] # 配体的原子数
    dm = distance_matrix(lig_coord, pock_coord) # 复合的距离矩阵
    lig_pock_dist, lig_pock_edge, node_map = edge_ligand_pocket(dm, lig_size, theta=theta, keep_pock=keep_pock) # line183的函数

    # construct ligand graph & pocket graph
    lig_size, lig_fea, lig_edge = cons_mol_graph(lig_edge, lig_fea) # line163的函数 ***************************************************
    pock_size, pock_fea, pock_edge = cons_mol_graph(pock_edge, pock_fea)
    
    # construct spatial context graph based on distance
    dm = distance_matrix(lig_coord, lig_coord)
    edges, lig_dist = cons_spatial_gragh(dm, theta=theta) # line156的函数
    if pocket_spatial: # True
        dm_pock = distance_matrix(pock_coord, pock_coord)
        edges_pock, pock_dist = cons_spatial_gragh(dm_pock, theta=theta)
    lig_edge = edges
    pock_edge = edges_pock
    
    # map new pocket graph
    pock_size = len(node_map)
    pock_fea = pock_fea[sorted(node_map.keys())]
    pock_edge, pock_dist = pocket_subgraph(node_map, pock_edge, pock_dist) # line168的函数
    pock_coord_ = pock_coord[sorted(node_map.keys())]
    
    # construct ligand-pocket graph
    size = lig_size + pock_size # 大小相加
    lig_fea, pock_fea = add_identity_fea(lig_fea, pock_fea, comb=add_fea) 

    feas = np.vstack([lig_fea, pock_fea]) if len(pock_fea) > 0 else lig_fea # 特征相融合 ***********************************************
    edges = lig_edge + lig_pock_edge + pock_edge # 邻接矩阵融合 ************************************************************************
    lig_atoms = get_lig_atom_types(feas) # line147的函数
    pock_atoms = get_pock_atom_types(feas) # line151的函数
    assert len(lig_atoms) ==  lig_size and len(pock_atoms) == pock_size
    
    atoms = np.concatenate([lig_atoms, pock_atoms]) if len(pock_fea) > 0 else lig_atoms # 原子融合
    
    lig_atoms_raw = np.array(lig_atoms_raw)
    pock_atoms_raw = np.array(pock_atoms_raw)
    pock_atoms_raw = pock_atoms_raw[sorted(node_map.keys())]
    atoms_raw = np.concatenate([lig_atoms_raw, pock_atoms_raw]) if len(pock_atoms_raw) > 0 else lig_atoms_raw # 原始原子融合 **********
     
    coords = np.vstack([lig_coord, pock_coord_]) if len(pock_fea) > 0 else lig_coord # 坐标融合 ***************************************
    if len(pock_fea) > 0:
        assert size==max(node_map.values())+1
    assert feas.shape[0]==coords.shape[0]
    
    # construct 3d edge_feat
    dist_mat = distance.cdist(coords, coords, 'euclidean')
    np.fill_diagonal(dist_mat, np.inf)
    dist_graph_base = dist_mat.copy()
    dist_feat = dist_graph_base[dist_graph_base < theta].reshape(-1,1)
    edge_index_i = np.array([i[0] for i in edges])[np.newaxis,:]
    edge_index_j = np.array([j[1] for j in edges])[np.newaxis,:]
    edge_index = np.concatenate([edge_index_i, edge_index_j], axis=0)
    edge_feat_3d = get_3d_feature(edge_index, coords)
    edge_feat_3d[np.isinf(edge_feat_3d)] = np.nan
    edge_feat_3d[np.isnan(edge_feat_3d)] = 0
    edge_feat = np.hstack((edge_feat_3d, dist_feat))
    # return lig_size, coords, feas, edges, atoms_raw # 返回配体大小，坐标，特征，邻接矩阵，原始原子
    
    return feas, edge_index, edge_feat

def random_split(dataset_size, split_ratio=1, seed=0, shuffle=True):
    """random splitter"""
    np.random.seed(seed)
    indices = list(range(dataset_size))
    np.random.shuffle(indices)
    split = int(split_ratio * dataset_size)
    train_idx, valid_idx = indices[:split], indices[split:]
    return train_idx, valid_idx


def process_dataset(protein_name, core_lst, path, cutoff):
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
        
    refined_id, refined_data, refined_pk = [], [], []
    core_id, core_data, core_pk = [], [], []
    fault=[]
    for k, v in data_dict.items():
        ligand = (v['lig_fea'], v['lig_co'], v['lig_atoms'], v['lig_eg'])
        pocket = (v['pock_fea'], v['pock_co'], v['pock_atoms'], v['pock_eg'])
        try:
            graph = cons_lig_pock_graph_with_spatial_context(ligand, pocket, add_fea=3, theta=cutoff, keep_pock=False, pocket_spatial=True) ######在这个函数里把dataset.py里的移过去就好了？
            cofeat, pk = v['type_pair'], v['pk']
            graph = list(graph) + [cofeat]
            if k in core_lst:
                core_id.append(k)
                core_data.append(graph)
                core_pk.append(pk)
                continue
            refined_id.append(k)
            refined_data.append(graph)
            refined_pk.append(pk)
        except:
            fault.append(k)
    # split train and valid
    train_idxs, valid_idxs = random_split(len(refined_data), split_ratio=1, seed=2020, shuffle=True)
    train_i = [refined_id[i] for i in train_idxs]
    train_g = [refined_data[i] for i in train_idxs]
    train_y = [refined_pk[i] for i in train_idxs]
    valid_i = [refined_id[i] for i in valid_idxs]
    valid_g = [refined_data[i] for i in valid_idxs]
    valid_y = [refined_pk[i] for i in valid_idxs]
    train = (train_i, train_g, train_y)
    valid = (valid_i, valid_g, valid_y)
    test = (core_id, core_data, core_pk)
    return train, valid, test

def write_pickle(data, output_path, dataset_name):
    train = []
    valid = []
    test = []
    for i in data:
        train.append(i[0])
        valid.append(i[1])
        test.append(i[2])  
        
    with open(os.path.join(output_path, dataset_name + '_train.pkl'), 'wb') as f:
        pickle.dump(train, f)
    with open(os.path.join(output_path, dataset_name + '_val.pkl'), 'wb') as f:
        pickle.dump(valid, f)
    with open(os.path.join(output_path, dataset_name + '_test.pkl'), 'wb') as f:
        pickle.dump(test, f)
    
def pocket_mol2(path):
     for filename in os.listdir(path):
         pocket_pdb=os.path.join(path,filename,str(filename)+'_pocket.pdb')
         pocket_mol2=os.path.join(path,filename,str(filename)+'_pocket.mol2')
         cmd='obabel -ipdb '+str(pocket_pdb)+' -omol2 -O '+str(pocket_mol2)
         os.system(cmd)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path_core', type=str, default='../data/coreset')
    parser.add_argument('--data_path_refined', type=str, default='../data/refined-set')
    parser.add_argument('--output_path', type=str, default='../data/')
    parser.add_argument('--dataset_name', type=str, default='pdbbind2016')
    parser.add_argument('--n_jobs', type=int, default=-1)
    parser.add_argument('--cutoff', type=float, default=5.5)
    parser.add_argument('--file_process', type=bool, default=False)
    args = parser.parse_args()
    
    if args.file_process:
        print('processing file')
        pocket_mol2(args.data_path_core)
        pocket_mol2(args.data_path_refined)
        print('done')
    else:
        print('ignore file process')
    
    core_set_list = [x for x in os.listdir(args.data_path_core) if len(x) == 4]
    refined_set_list = [x for x in os.listdir(args.data_path_refined) if len(x) == 4]
    
    print('processing dataset')
    data = pmap_multi(process_dataset, zip(refined_set_list), 
                      n_jobs=args.n_jobs, 
                      desc='Get receptors', 
                      core_lst=core_set_list,
                      path=args.data_path_refined,
                      cutoff=args.cutoff)
    write_pickle(data, args.output_path, args.dataset_name)
    print('done')
