import argparse
import os
import pickle
from functools import partial

import numpy as np
import openbabel
from openbabel import pybel
from scipy.spatial import distance, distance_matrix
from utils import *

from FGNN.process.featurizer import Featurizer


class Featurizer():
    """Calcaulates atomic features for molecules. Features can encode atom type,
    native pybel properties or any property defined with SMARTS patterns
    Attributes
    ----------
    FEATURE_NAMES: list of strings
        Labels for features (in the same order as features)
    NUM_ATOM_CLASSES: int
        Number of atom codes
    ATOM_CODES: dict
        Dictionary mapping atomic numbers to codes
    NAMED_PROPS: list of string
        Names of atomic properties to retrieve from pybel.Atom object
    CALLABLES: list of callables
        Callables used to calculcate custom atomic properties
    SMARTS: list of SMARTS strings
        SMARTS patterns defining additional atomic properties
    """ 

    def __init__(self, atom_codes=None, atom_labels=None,
                 named_properties=None, save_molecule_codes=True,
                 custom_properties=None, smarts_properties=None,
                 smarts_labels=None):

        """Creates Featurizer with specified types of features. Elements of a
        feature vector will be in a following order: atom type encoding
        (defined by atom_codes), Pybel atomic properties (defined by
        named_properties), molecule code (if present), custom atomic properties
        (defined `custom_properties`), and additional properties defined with
        SMARTS (defined with `smarts_properties`).
        Parameters
        ----------
        atom_codes: dict, optional
            Dictionary mapping atomic numbers to codes. It will be used for
            one-hot encoging therefore if n different types are used, codes
            shpuld be from 0 to n-1. Multiple atoms can have the same code,
            e.g. you can use {6: 0, 7: 1, 8: 1} to encode carbons with [1, 0]
            and nitrogens and oxygens with [0, 1] vectors. If not provided,
            default encoding is used.
        atom_labels: list of strings, optional
            Labels for atoms codes. It should have the same length as the
            number of used codes, e.g. for `atom_codes={6: 0, 7: 1, 8: 1}` you
            should provide something like ['C', 'O or N']. If not specified
            labels 'atom0', 'atom1' etc are used. If `atom_codes` is not
            specified this argument is ignored.
        named_properties: list of strings, optional
            Names of atomic properties to retrieve from pybel.Atom object. If
            not specified ['hyb', 'heavyvalence', 'heterovalence',
            'partialcharge'] is used.
        save_molecule_codes: bool, optional (default True)
            If set to True, there will be an additional feature to save
            molecule code. It is usefeul when saving molecular complex in a
            single array.
        custom_properties: list of callables, optional
            Custom functions to calculate atomic properties. Each element of
            this list should be a callable that takes pybel.Atom object and
            returns a float. If callable has `__name__` property it is used as
            feature label. Otherwise labels 'func<i>' etc are used, where i is
            the index in `custom_properties` list.
        smarts_properties: list of strings, optional
            Additional atomic properties defined with SMARTS patterns. These
            patterns should match a single atom. If not specified, deafult
            patterns are used.
        smarts_labels: list of strings, optional
            Labels for properties defined with SMARTS. Should have the same
            length as `smarts_properties`. If not specified labels 'smarts0',
            'smarts1' etc are used. If `smarts_properties` is not specified
            this argument is ignored.
        """
        # atom_codes（dict），atom_labels，named_properties，save_molecule_codes，custom_properties，smarts_properties，smarts_labels
        # Remember namse of all features in the correct order
        self.FEATURE_NAMES = []

        if atom_codes is not None: 
            if not isinstance(atom_codes, dict):
                raise TypeError('Atom codes should be dict, got %s instead'
                                % type(atom_codes))
            codes = set(atom_codes.values()) 
            for i in range(len(codes)): 
                if i not in codes:
                    raise ValueError('Incorrect atom code %s' % i)

            self.NUM_ATOM_CLASSES = len(codes) 
            self.ATOM_CODES = atom_codes 
            if atom_labels is not None: 
                if len(atom_labels) != self.NUM_ATOM_CLASSES:
                    raise ValueError('Incorrect number of atom labels: '
                                     '%s instead of %s'
                                     % (len(atom_labels), self.NUM_ATOM_CLASSES))
            else:
                atom_labels = ['atom%s' % i for i in range(self.NUM_ATOM_CLASSES)] 
            self.FEATURE_NAMES += atom_labels 
        else: 
            self.ATOM_CODES = {}

            metals = ([3, 4, 11, 12, 13] + list(range(19, 32))
                      + list(range(37, 51)) + list(range(55, 84))
                      + list(range(87, 104))) 

            # List of tuples (atomic_num, class_name) with atom types to encode.
            atom_classes = [
                (5, 'B'),
                (6, 'C'),
                (7, 'N'),
                (8, 'O'),
                (15, 'P'),
                (16, 'S'),
                (34, 'Se'),
                ([9, 17, 35, 53], 'halogen'),
                (metals, 'metal')
            ] 

            for code, (atom, name) in enumerate(atom_classes): 
                if type(atom) is list: 
                    for a in atom:
                        self.ATOM_CODES[a] = code 
                else:
                    self.ATOM_CODES[atom] = code 
                self.FEATURE_NAMES.append(name) 

            self.NUM_ATOM_CLASSES = len(atom_classes) 

        if named_properties is not None: 
            if not isinstance(named_properties, (list, tuple, np.ndarray)):
                raise TypeError('named_properties must be a list')
            allowed_props = [prop for prop in dir(pybel.Atom)
                             if not prop.startswith('__')] 
            for prop_id, prop in enumerate(named_properties):
                if prop not in allowed_props:
                    raise ValueError(
                        'named_properties must be in pybel.Atom attributes,'
                        ' %s was given at position %s' % (prop_id, prop)
                    )
            self.NAMED_PROPS = named_properties
        else: 
            # pybel.Atom properties to save
            self.NAMED_PROPS = ['hyb', 'heavydegree', 'heterodegree',
                                'partialcharge']
        self.FEATURE_NAMES += self.NAMED_PROPS 

        if not isinstance(save_molecule_codes, bool): 
            raise TypeError('save_molecule_codes should be bool, got %s '
                            'instead' % type(save_molecule_codes))
        self.save_molecule_codes = save_molecule_codes # True
        if save_molecule_codes:
            # Remember if an atom belongs to the ligand or to the protein
            self.FEATURE_NAMES.append('molcode') 

        self.CALLABLES = []
        if custom_properties is not None: 
            for i, func in enumerate(custom_properties): 
                if not callable(func): 
                    raise TypeError('custom_properties should be list of'
                                    ' callables, got %s instead' % type(func))
                name = getattr(func, '__name__', '')
                if name == '':
                    name = 'func%s' % i
                self.CALLABLES.append(func)
                self.FEATURE_NAMES.append(name) 

        if smarts_properties is None: 
            # SMARTS definition for other properties
            self.SMARTS = [
                '[#6+0!$(*~[#7,#8,F]),SH0+0v2,s+0,S^3,Cl+0,Br+0,I+0]',
                '[a]',
                '[!$([#1,#6,F,Cl,Br,I,o,s,nX3,#7v5,#15v5,#16v4,#16v6,*+1,*+2,*+3])]',
                '[!$([#6,H0,-,-2,-3]),$([!H0;#7,#8,#9])]',
                '[r]'
            ]
            smarts_labels = ['hydrophobic', 'aromatic', 'acceptor', 'donor',
                             'ring']
        elif not isinstance(smarts_properties, (list, tuple, np.ndarray)): 
            raise TypeError('smarts_properties must be a list')
        else:
            self.SMARTS = smarts_properties

        if smarts_labels is not None: 
            if len(smarts_labels) != len(self.SMARTS): 
                raise ValueError('Incorrect number of SMARTS labels: %s'
                                 ' instead of %s'
                                 % (len(smarts_labels), len(self.SMARTS)))
        else:
            smarts_labels = ['smarts%s' % i for i in range(len(self.SMARTS))]

        # Compile patterns
        self.compile_smarts()
        self.FEATURE_NAMES += smarts_labels 

    def compile_smarts(self):
        self.__PATTERNS = []
        for smarts in self.SMARTS:
            self.__PATTERNS.append(pybel.Smarts(smarts))

    def encode_num(self, atomic_num): 
        """Encode atom type with a binary vector. If atom type is not included in
        the `atom_classes`, its encoding is an all-zeros vector.
        Parameters
        ----------
        atomic_num: int
            Atomic number
        Returns
        -------
        encoding: np.ndarray
            Binary vector encoding atom type (one-hot or null).
        """

        if not isinstance(atomic_num, int): 
            raise TypeError('Atomic number must be int, %s was given'
                            % type(atomic_num))

        encoding = np.zeros(self.NUM_ATOM_CLASSES) 
        try:
            encoding[self.ATOM_CODES[atomic_num]] = 1.0
        except:
            pass
        return encoding 

    def find_smarts(self, molecule):
        """Find atoms that match SMARTS patterns.
        Parameters
        ----------
        molecule: pybel.Molecule
        Returns
        -------
        features: np.ndarray
            NxM binary array, where N is the number of atoms in the `molecule`
            and M is the number of patterns. `features[i, j]` == 1.0 if i'th
            atom has j'th property
        """

        if not isinstance(molecule, pybel.Molecule): 
            raise TypeError('molecule must be pybel.Molecule object, %s was given'
                            % type(molecule))

        features = np.zeros((len(molecule.atoms), len(self.__PATTERNS))) 

        for (pattern_id, pattern) in enumerate(self.__PATTERNS): 
            atoms_with_prop = np.array(list(*zip(*pattern.findall(molecule))),
                                       dtype=int) - 1 
            features[atoms_with_prop, pattern_id] = 1.0 
        return features 

    def get_features(self, molecule, molcode=None): 
        """Get coordinates and features for all heavy atoms in the molecule.
        Parameters
        ----------
        molecule: pybel.Molecule
        molcode: float, optional
            Molecule type. You can use it to encode whether an atom belongs to
            the ligand (1.0) or to the protein (-1.0) etc.
        Returns
        -------
        coords: np.ndarray, shape = (N, 3)
            Coordinates of all heavy atoms in the `molecule`.
        features: np.ndarray, shape = (N, F)
            Features of all heavy atoms in the `molecule`: atom type
            (one-hot encoding), pybel.Atom attributes, type of a molecule
            (e.g protein/ligand distinction), and other properties defined with
            SMARTS patterns
        """

        if not isinstance(molecule, pybel.Molecule): 
            raise TypeError('molecule must be pybel.Molecule object,'
                            ' %s was given' % type(molecule))
        if molcode is None: 
            if self.save_molecule_codes is True:
                raise ValueError('save_molecule_codes is set to True,'
                                 ' you must specify code for the molecule')
        elif not isinstance(molcode, (float, int)):
            raise TypeError('motlype must be float, %s was given'
                            % type(molcode))

        coords = []
        features = []
        heavy_atoms = []

        for i, atom in enumerate(molecule): 
            # ignore hydrogens and dummy atoms (they have atomicnum set to 0)
            if atom.atomicnum > 1: 
                heavy_atoms.append(i) 
                coords.append(atom.coords) 

                features.append(np.concatenate((
                    self.encode_num(atom.atomicnum),
                    [atom.__getattribute__(prop) for prop in self.NAMED_PROPS],
                    [func(atom) for func in self.CALLABLES],
                ))) 

        coords = np.array(coords, dtype=np.float32)
        features = np.array(features, dtype=np.float32) 
        if self.save_molecule_codes:
            features = np.hstack((features,
                                  molcode * np.ones((len(features), 1)))) 
        features = np.hstack([features,
                              self.find_smarts(molecule)[heavy_atoms]]) 

        if np.isnan(features).any(): 
            raise RuntimeError('Got NaN when calculating features')

        return coords, features 

    def to_pickle(self, fname='featurizer.pkl'): 
        """Save featurizer in a given file. Featurizer can be restored with
        `from_pickle` method.
        Parameters
        ----------
        fname: str, optional
           Path to file in which featurizer will be saved
        """

        # patterns can't be pickled, we need to temporarily remove them
        patterns = self.__PATTERNS[:]
        del self.__PATTERNS
        try:
            with open(fname, 'wb') as f:
                pickle.dump(self, f)
        finally:
            self.__PATTERNS = patterns[:]

    @staticmethod
    def from_pickle(fname): 
        """Load pickled featurizer from a given file
        Parameters
        ----------
        fname: str, optional
           Path to file with saved featurizer
        Returns
        -------
        featurizer: Featurizer object
           Loaded featurizer
        """
        with open(fname, 'rb') as f:
            featurizer = pickle.load(f)
        featurizer.compile_smarts()
        return featurizer

def rbf_distance_featurizer(dist_list, divisor=1):                                                                                                                                                        
    # you want to use a divisor that is close to 4/7 times the average distance that you want to encode                                                                                                                   
    length_scale_list = [1.5 ** x for x in range(15)]                                                                                                                                                                     
    center_list = [0. for _ in range(15)]                                                                                                                                                                                 
                                                                                                                                                                                                                          
    num_edge = len(dist_list)                                                                                                                                                                                             
    dist_list = np.array(dist_list)                                                                                                                                                                                       
                                                                                                                                                                                                                          
    transformed_dist = [np.exp(- ((dist_list / divisor) ** 2) / float(length_scale))                                                                                                                                      
                        for length_scale, center in zip(length_scale_list, center_list)]                                                                                                                                  
                                                                                                                                                                                                                          
    transformed_dist = np.array(transformed_dist).T                                                                                                                                                                       
    transformed_dist = transformed_dist.reshape((num_edge, -1))                                                                                                                                                           
    return torch.from_numpy(transformed_dist.astype(np.float32))                                                                                                                                                          
                                                                                                                                                                                                                          
def pocket_atom_num_from_mol2(name, path):                                                                                                                                                                  
    n = 0                                                                                                                                                                                                                 
    with open('%s/%s/%s_pocket.mol2' % (path, name, name)) as f:                                                                                                                                        
        for line in f:                                                                                                                                                                                                    
            if '<TRIPOS>ATOM' in line:                                                                                                                                                                            
                break                                                                                                                                                                                                     
        for line in f:                                                                                                                                                                                                    
            cont = line.split()
            try:                                                                                                                                                                                          
                if '<TRIPOS>BOND' in line or cont[7] == 'HOH':                                                                                                                                                   
                    break                                                                                                                                                                                                     
                n += int(cont[5][0] != 'H') 
            except:
                print(name)                                                                                                                                                                         
    return n                                                                                                                                                                                           
                                                                                                                                                                                                                          
def pocket_atom_num_from_pdb(name, path):                                                                                                                                                                 
    n = 0                                                                                                                                                                                                                 
    with open('%s/%s/%s_pocket.pdb' % (path, name, name)) as f:                                                                                                                                          
        for line in f:                                                                                                                                                                                                    
            if 'REMARK' in line:                                                                                                                                                                              
                break                                                                                                                                                                                                     
        for line in f:                                                                                                                                                                                                    
            cont = line.split()                                                                                                                                                                                           
            # break                                                                                                                                                                                                       
            if cont[0] == 'CONECT':                                                                                                                                                                            
                break                                                                                                                                                                                                     
            n += int(cont[-1] != 'H' and cont[0] == 'ATOM')                                                                                                                                                
    return n                                                                                                                                                                                           
                                                                                                                                                                                                                          
## function -- feature                                                                                                                                                                                                    
def gen_feature(path, name, featurizer):                                                                                                                                                   
    charge_idx = featurizer.FEATURE_NAMES.index('partialcharge')                                                                                                                                                          
    ligand = next(pybel.readfile('mol2', '%s/%s/%s_ligand.mol2' % (path, name, name)))                                                                                                                  
    ligand_coords, ligand_features = featurizer.get_features(ligand, molcode=1)                                                                                                   
    pocket = next(pybel.readfile('mol2' ,'%s/%s/%s_pocket.mol2' % (path, name, name)))                                                                                                                   
    pocket_coords, pocket_features = featurizer.get_features(pocket, molcode=-1)                                                                                                                              
    node_num = pocket_atom_num_from_mol2(name, path)                                                                                                                                                             
    pocket_coords = pocket_coords[:node_num]                                                                                                                                                                     
    pocket_features = pocket_features[:node_num]                                                                                                                                                                
    try:                                                                                                                                                                                                                  
        assert (ligand_features[:, charge_idx] != 0).any()                                                                                                                                                
        assert (pocket_features[:, charge_idx] != 0).any()                                                                                                                                                            
        assert (ligand_features[:, :9].sum(1) != 0).all()                                                                                                                                   
    except:                                                                                                                                                                                                               
        print(name)                                                                                                                                                                                                       
    lig_atoms, pock_atoms = [], []                                                                                                                                                                                        
    for i, atom in enumerate(ligand):                                                                                                                                                                                     
        if atom.atomicnum > 1:                                                                                                                                                                
            lig_atoms.append(atom.atomicnum)                                                                                                                                                                 
    for i, atom in enumerate(pocket):                                                                                                                                                                                     
        if atom.atomicnum > 1:                                                                                                                                                                                            
            pock_atoms.append(atom.atomicnum)                                                                                                                                                                  
    for x in pock_atoms[node_num:]:
        try:                                                                                                                                                                                       
            assert x == 8
        except:
            print(name)                                                                                                                                                                                                     
    pock_atoms = pock_atoms[:node_num]                                                                                                                                                                    
    assert len(lig_atoms)==len(ligand_features) and len(pock_atoms)==len(pocket_features)                                                                                                         
                                                                                                                                                                                                                          
    ligand_edges = gen_pocket_graph(ligand)                                                                                                                                                                      
    pocket_edges = gen_pocket_graph(pocket)                                                                                                                                                                      
    # print(pocket_edges)                                                                                                                                                                                                 
    return {'lig_co': ligand_coords, 'lig_fea': ligand_features, 'lig_atoms': lig_atoms, 'lig_eg': ligand_edges, 'pock_co': pocket_coords, 'pock_fea': pocket_features, 'pock_atoms': pock_atoms, 'pock_eg': pocket_edges}
                                                                                                                                                                                             
## function -- pocket graph                                                                                                                                                                                               
def gen_pocket_graph(pocket):                                                                                                                                                                               
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
    return edge_l_new                                                                                                                                                                                             
                                                                                                                                                                                                                          
def dist_filter(dist_matrix, theta):                                                                                                                                                                          
    pos = np.where(dist_matrix<=theta)                                                                                                                                                                                    
    ligand_list, pocket_list = pos                                                                                                                                                                                        
    return ligand_list, pocket_list                                                                                                                                                                      
                                                                                                                                                                                                                          
def pairwise_atomic_types(path, processed_dict, atom_types, atom_types_):                                                                                                                       
    keys = [(i,j) for i in atom_types_ for j in atom_types]                                                                                                                                                 
    # for name in os.listdir(path):                                                                                                                                                                                       
    #     if len(name) != 4:                                                                                                                                                                                              
    #         continue                                                                                                                                                                                                    
    name = list(processed_dict.keys())[0]                                                                                                                                                                                 
    ligand = next(pybel.readfile('mol2', '%s/%s/%s_ligand.mol2' % (path, name, name)))                                                                                                              
    pocket = next(pybel.readfile('pdb' ,'%s/%s/%s_protein.pdb' % (path, name, name)))                                                                                                                 
    coords_lig = np.vstack([atom.coords for atom in ligand])                                                                                                                                                       
    coords_poc = np.vstack([atom.coords for atom in pocket])                                                                                                                                                       
    atom_map_lig = [atom.atomicnum for atom in ligand]                                                                                                                                                         
    atom_map_poc = [atom.atomicnum for atom in pocket]                                                                                                                                                         
    dm = distance_matrix(coords_lig, coords_poc)                                                                                                                                                           
    ligs, pocks = dist_filter(dm, 12)                                                                                                                                                                      
                                                                                                                                                                                                                          
    fea_dict = {k: 0 for k in keys}                                                                                                                                                                       
    for x, y in zip(ligs, pocks):                                                                                                                                                                                         
        x, y = atom_map_lig[x], atom_map_poc[y]                                                                                                                                                           
        if x not in atom_types or y not in atom_types_: continue                                                                                                                                               
        fea_dict[(y, x)] += 1                                                                                                                                                                                 
    processed_dict[name]['type_pair'] = list(fea_dict.values())                                                                                                                       
                                                                                                                                                                                                                          
    return processed_dict                                                                                                                                                                                        
                                                                                                                                                                                                                          
def load_pk_data(data_path):                                                                                                                                                                                       
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
    return res                                                                                                                                                                                    
                                                                                                                                                                                                                          
def get_lig_atom_types(feat):                                                                                                                                                                                             
    pos = np.where(feat[:,:9]>0)                                                                                                                                                                                          
    src_list, dst_list = pos                                                                                                                                                                                              
    return dst_list                                                                                                                                                                                                       
                                                                                                                                                                                                                          
def get_pock_atom_types(feat):                                                                                                                                                                                            
    pos = np.where(feat[:,18:27]>0)                                                                                                                                                                                       
    src_list, dst_list = pos                                                                                                                                                                                              
    return dst_list                                                                                                                                                                                                       
                                                                                                                                                                                                                          
def cons_spatial_gragh(dist_matrix, theta=5):                                                                                                                                                            
    pos = np.where((dist_matrix<=theta)&(dist_matrix!=0))                                                                                                                                                                 
    src_list, dst_list = pos                                                                                                                                                                                              
    dist_list = dist_matrix[pos]                                                                                                                                                                                          
    edges = [(x,y) for x,y in zip(src_list, dst_list)]                                                                                                                                                                    
    return edges, dist_list                                                                                                                                                                                 
                                                                                                                                                                                                                          
def cons_mol_graph(edges, feas):                                                                                                                                                                        
    size = feas.shape[0]                                                                                                                                                                                                  
    edges = [(x,y) for x,y,t in edges]                                                                                                                                                                                    
    return size, feas, edges                                                                                                                                                                               
                                                                                                                                                                                                                          
def pocket_subgraph(node_map, edge_list, pock_dist):                                                                                                                                                              
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
    return edge_l, dist_l                                                                                                                                                                                    
                                                                                                                                                                                                                          
def edge_ligand_pocket(dist_matrix, lig_size, theta=4, keep_pock=False, reset_idx=True):                                                                                                         
                                                                                                                                                                                                                          
    pos = np.where(dist_matrix<=theta)                                                                                                                                                                         
    ligand_list, pocket_list = pos                                                                                                                                                                                        
    if keep_pock: # False                                                                                                                                                                                                 
        node_list = range(dist_matrix.shape[1])                                                                                                                                                                           
    else:                                                                                                                                                                                                                 
        node_list = sorted(list(set(pocket_list)))                                                                                                                                                      
    node_map = {node_list[i]:i+lig_size for i in range(len(node_list))}                                                                                
                                                                                                                                                                                                                          
    dist_list = dist_matrix[pos]                                                                                                                                                                       
    if reset_idx: # True                                                                                                                                                                                                  
        edge_list = [(x,node_map[y]) for x,y in zip(ligand_list, pocket_list)]                                                                                                                               
    else:                                                                                                                                                                                                                 
        edge_list = [(x,y) for x,y in zip(ligand_list, pocket_list)]                                                                                                                                                      
                                                                                                                                                                                                                          
    edge_list += [(y,x) for x,y in edge_list]                                                                                                                                                                   
    dist_list = np.concatenate([dist_list, dist_list])                                                                                                                                     
                                                                                                                                                                                                                          
    return dist_list, edge_list, node_map                                                                                                                                                               
                                                                                                                                                                                                                          
def add_identity_fea(lig_fea, pock_fea, comb=1):                                                                                                                                                 
    if comb == 1:                                                                                                                                                                                                  
        lig_fea = np.hstack([lig_fea, [[1]]*len(lig_fea)])                                                                                                                                                     
        pock_fea = np.hstack([pock_fea, [[-1]]*len(pock_fea)])                                                                                                                                             
    elif comb == 2:                                                                                                                                                                                                       
        lig_fea = np.hstack([lig_fea, [[1,0]]*len(lig_fea)])                                                                                                                                                              
        pock_fea = np.hstack([pock_fea, [[0,1]]*len(pock_fea)])                                                                                                                                                           
    else:                                                                                                                                                                                                                 
        lig_fea = np.hstack([lig_fea, [[0]*lig_fea.shape[1]]*len(lig_fea)])                                                                                                                                               
        if len(pock_fea) > 0:                                                                                                                                                                                             
            pock_fea = np.hstack([[[0]*pock_fea.shape[1]]*len(pock_fea), pock_fea])                                                                                                                                       
                                                                                                                                                                                                                          
    return lig_fea, pock_fea                                                                                                                                                                          
                                                                                                                                                                                                                          
def get_neighbors(one_hot_src, dst_idx):                                                                                                                                                                                  
    select_idx = np.nonzero(one_hot_src)                                                                                                                                                                                  
    neighbors = dst_idx[select_idx]                                                                                                                                                                                       
    return neighbors.tolist()                                                                                                                                                                                             
                                                                                                                                                                                                                          
def D3_info(a, b, c):                                                                                                                                                                                                                                                                                                                                                                                                                     
    ab = b - a                                                                                                                                                                                                     
    ac = c - a                                                                                                                                                                                                      
    cosine_angle = np.dot(ab, ac) / (np.linalg.norm(ab) * np.linalg.norm(ac))                                                                                                                                             
    cosine_angle = cosine_angle if cosine_angle >= -1.0 else -1.0                                                                                                                                                         
    angle = np.arccos(cosine_angle)                                                                                                                                                                                       
                                                                                                                                                                                                                   
    ab_ = np.sqrt(np.sum(ab ** 2))                                                                                                                                                                                        
    ac_ = np.sqrt(np.sum(ac ** 2))                                                                                                                                                                                  
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
        neighbors_ls.append(tmp)                                                                                                                           
    edge_feature = list(map(partial(D3_info_cal, coords=coords), neighbors_ls))                                                                                                                                           
    return np.array(edge_feature)                                                                                                                                                                                         
                                                                                                                                                                                                                          
def cons_lig_pock_graph_with_spatial_context(ligand, pocket, add_fea=2, theta=5, keep_pock=False, pocket_spatial=True):                                                                        
    lig_fea, lig_coord, lig_atoms_raw, lig_edge = ligand                                                                                                                                                   
    pock_fea, pock_coord, pock_atoms_raw, pock_edge = pocket                                                                                                                                                              
                                                                                                                                                                                                                          
    # inter-relation between ligand and pocket                                                                                                                                                                            
    lig_size = lig_fea.shape[0]                                                                                                                                                                                   
    dm = distance_matrix(lig_coord, pock_coord)                                                                                                                                                                  
    lig_pock_dist, lig_pock_edge, node_map = edge_ligand_pocket(dm, lig_size, theta=theta, keep_pock=keep_pock)                                                                                              
                                                                                                                                                                                                                          
    # construct ligand graph & pocket graph                                                                                                                                                                               
    lig_size, lig_fea, lig_edge = cons_mol_graph(lig_edge, lig_fea)                                                                                       
    pock_size, pock_fea, pock_edge = cons_mol_graph(pock_edge, pock_fea)                                                                                                                                                  
                                                                                                                                                                                                                          
    # construct spatial context graph based on distance                                                                                                                                                                   
    dm = distance_matrix(lig_coord, lig_coord)                                                                                                                                                                            
    edges, lig_dist = cons_spatial_gragh(dm, theta=theta)                                                                                                                                                     
    if pocket_spatial: # True                                                                                                                                                                                             
        dm_pock = distance_matrix(pock_coord, pock_coord)                                                                                                                                                                 
        edges_pock, pock_dist = cons_spatial_gragh(dm_pock, theta=theta)                                                                                                                                                  
    lig_edge = edges                                                                                                                                                                                                      
    pock_edge = edges_pock                                                                                                                                                                                                
                                                                                                                                                                                                                          
    # map new pocket graph                                                                                                                                                                                                
    pock_size = len(node_map)                                                                                                                                                                                             
    pock_fea = pock_fea[sorted(node_map.keys())]                                                                                                                                                                          
    pock_edge, pock_dist = pocket_subgraph(node_map, pock_edge, pock_dist)                                                                                                                                   
    pock_coord_ = pock_coord[sorted(node_map.keys())]                                                                                                                                                                     
                                                                                                                                                                                                                          
    # construct ligand-pocket graph                                                                                                                                                                                       
    size = lig_size + pock_size                                                                                                                                                                                     
    lig_fea, pock_fea = add_identity_fea(lig_fea, pock_fea, comb=add_fea)                                                                                                                                                 
                                                                                                                                                                                                                          
    feas = np.vstack([lig_fea, pock_fea]) if len(pock_fea) > 0 else lig_fea                                                                                        
    edges = lig_edge + lig_pock_edge + pock_edge                                                                                       
    lig_atoms = get_lig_atom_types(feas)                                                                                                                                                                      
    pock_atoms = get_pock_atom_types(feas)                                                                                                                                                                    
    assert len(lig_atoms) ==  lig_size and len(pock_atoms) == pock_size                                                                                                                                                   
                                                                                                                                                                                                                          
    atoms = np.concatenate([lig_atoms, pock_atoms]) if len(pock_fea) > 0 else lig_atoms                                                                                                                             
                                                                                                                                                                                                                          
    lig_atoms_raw = np.array(lig_atoms_raw)                                                                                                                                                                               
    pock_atoms_raw = np.array(pock_atoms_raw)                                                                                                                                                                             
    pock_atoms_raw = pock_atoms_raw[sorted(node_map.keys())]                                                                                                                                                              
    atoms_raw = np.concatenate([lig_atoms_raw, pock_atoms_raw]) if len(pock_atoms_raw) > 0 else lig_atoms_raw                                                                                          
                                                                                                                                                                                                                          
    coords = np.vstack([lig_coord, pock_coord_]) if len(pock_fea) > 0 else lig_coord                                                                                        
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
    edge_feat_rbf = rbf_distance_featurizer(dist_feat)                                                                                                                                                                    
    # return lig_size, coords, feas, edges, atoms_raw                                                                                                                                             
                                                                                                                                                                                                                          
    return feas, coords, edge_index, edge_feat, edge_feat_rbf    