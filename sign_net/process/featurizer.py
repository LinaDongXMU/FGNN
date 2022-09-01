import pickle
import numpy as np
from openbabel import pybel

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
    """ # NUM_ATOM_CLASSES是原子类数（int），ATOM_CODES将原子数映射到代码的字典（dict），NAMED_PROPS原子性质名字的列表（储存str的list），CALLABLES用于计算自定义原子属性的可调用函数的列表（list），SMARTS模式定义其他原子属性的列表（list）

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

        if atom_codes is not None: # atom_codes非空的条件下，判断atom_codes的类型，不是字典报错
            if not isinstance(atom_codes, dict):
                raise TypeError('Atom codes should be dict, got %s instead'
                                % type(atom_codes))
            codes = set(atom_codes.values()) # codes是atom_codes的值组成的列表
            for i in range(len(codes)): # 值和长度应该是一致的，不然报错
                if i not in codes:
                    raise ValueError('Incorrect atom code %s' % i)

            self.NUM_ATOM_CLASSES = len(codes) # 原子种类数是编码列表的长度
            self.ATOM_CODES = atom_codes # 原子编码是一个字典
            if atom_labels is not None: # 原子标签的列表长度应该和原子种类数量相等，不然报错
                if len(atom_labels) != self.NUM_ATOM_CLASSES:
                    raise ValueError('Incorrect number of atom labels: '
                                     '%s instead of %s'
                                     % (len(atom_labels), self.NUM_ATOM_CLASSES))
            else:
                atom_labels = ['atom%s' % i for i in range(self.NUM_ATOM_CLASSES)] # 原子标签是[atom1,atom2,atom3,...,atomn]这种列表
            self.FEATURE_NAMES += atom_labels # 特征名列表等于原子标签列表不断相加
        else: # atom_codes若为空
            self.ATOM_CODES = {}

            metals = ([3, 4, 11, 12, 13] + list(range(19, 32))
                      + list(range(37, 51)) + list(range(55, 84))
                      + list(range(87, 104))) # 元素周期表中，所有金属元素的原子序号

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
            ] # 将原子种类分成这九类，它是一个由九个元组组成的列表

            for code, (atom, name) in enumerate(atom_classes): # 遍历atom_classes列表
                if type(atom) is list: # 卤素或金属是列表
                    for a in atom:
                        self.ATOM_CODES[a] = code # 遍历列表，a以f为例子，则9：7
                else:
                    self.ATOM_CODES[atom] = code # atom以c为例子，则6：2
                self.FEATURE_NAMES.append(name) # 特征名列表['B','C','N','O','P','S','Se','halogen','metal']

            self.NUM_ATOM_CLASSES = len(atom_classes) #原子数量的类别为9

        if named_properties is not None: # named_properties不为空的情况，判断named_properties的类型(list, tuple, np.ndarray)，不是报错
            if not isinstance(named_properties, (list, tuple, np.ndarray)):
                raise TypeError('named_properties must be a list')
            allowed_props = [prop for prop in dir(pybel.Atom)
                             if not prop.startswith('__')] # 允许性质列表，判断性质在不在这个列表中，不在报错
            for prop_id, prop in enumerate(named_properties):
                if prop not in allowed_props:
                    raise ValueError(
                        'named_properties must be in pybel.Atom attributes,'
                        ' %s was given at position %s' % (prop_id, prop)
                    )
            self.NAMED_PROPS = named_properties
        else: # 若named_properties为空
            # pybel.Atom properties to save
            self.NAMED_PROPS = ['hyb', 'heavydegree', 'heterodegree',
                                'partialcharge']
        self.FEATURE_NAMES += self.NAMED_PROPS # 特征名列表['B','C','N','O','P','S','Se','halogen','metal','hyb', 'heavydegree', 'heterodegree','partialcharge']

        if not isinstance(save_molecule_codes, bool): # 判断save_molecule_codes是否为布尔值，不是报错
            raise TypeError('save_molecule_codes should be bool, got %s '
                            'instead' % type(save_molecule_codes))
        self.save_molecule_codes = save_molecule_codes # True
        if save_molecule_codes:
            # Remember if an atom belongs to the ligand or to the protein
            self.FEATURE_NAMES.append('molcode') # 特征名列表['B','C','N','O','P','S','Se','halogen','metal','hyb', 'heavydegree', 'heterodegree','partialcharge','molcode']

        self.CALLABLES = []
        if custom_properties is not None: # 若custom_properties非空
            for i, func in enumerate(custom_properties): # 遍历custom_properties列表
                if not callable(func): # 函数不能调用报错
                    raise TypeError('custom_properties should be list of'
                                    ' callables, got %s instead' % type(func))
                name = getattr(func, '__name__', '')
                if name == '':
                    name = 'func%s' % i
                self.CALLABLES.append(func)
                self.FEATURE_NAMES.append(name) # 在特征里加上这个函数定义特征的名字

        if smarts_properties is None: # 若smarts_properties为空
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
        elif not isinstance(smarts_properties, (list, tuple, np.ndarray)): # 判断smarts_properties的类型(list, tuple, np.ndarray)，不是报错
            raise TypeError('smarts_properties must be a list')
        else:
            self.SMARTS = smarts_properties

        if smarts_labels is not None: # 若smarts_labels非空
            if len(smarts_labels) != len(self.SMARTS): # 长度不相等报错
                raise ValueError('Incorrect number of SMARTS labels: %s'
                                 ' instead of %s'
                                 % (len(smarts_labels), len(self.SMARTS)))
        else:
            smarts_labels = ['smarts%s' % i for i in range(len(self.SMARTS))]

        # Compile patterns
        self.compile_smarts()
        self.FEATURE_NAMES += smarts_labels # 特征名列表['B','C','N','O','P','S','Se','halogen','metal','hyb', 'heavydegree', 'heterodegree','partialcharge','molcode','hydrophobic', 'aromatic', 'acceptor', 'donor', 'ring']

    def compile_smarts(self):
        self.__PATTERNS = []
        for smarts in self.SMARTS:
            self.__PATTERNS.append(pybel.Smarts(smarts))

    def encode_num(self, atomic_num): # 输入原子数量
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

        if not isinstance(atomic_num, int): # 如果输入的原子数量不是整数，报错
            raise TypeError('Atomic number must be int, %s was given'
                            % type(atomic_num))

        encoding = np.zeros(self.NUM_ATOM_CLASSES) 
        try:
            encoding[self.ATOM_CODES[atomic_num]] = 1.0
        except:
            pass
        return encoding # 返回一个编码原子类型的one-hot矩阵

    def find_smarts(self, molecule): # 输入是一个分子
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

        if not isinstance(molecule, pybel.Molecule): # 判断输入的分子是不是pybel可识别的分子，不是报错
            raise TypeError('molecule must be pybel.Molecule object, %s was given'
                            % type(molecule))

        features = np.zeros((len(molecule.atoms), len(self.__PATTERNS))) # 初始化特征矩阵（原子数，模式数）

        for (pattern_id, pattern) in enumerate(self.__PATTERNS): # 遍历模式列表
            atoms_with_prop = np.array(list(*zip(*pattern.findall(molecule))),
                                       dtype=int) - 1 ## 这里大概是怎么处理了特征矩阵的？？？
            features[atoms_with_prop, pattern_id] = 1.0 # 对应位置的数改1.0
        return features # 返回特征矩阵

    def get_features(self, molecule, molcode=None): # 传入的是pybel形式的分子，输出重原子的坐标（N，3）和特征（N，F）
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

        if not isinstance(molecule, pybel.Molecule): # 确定输入是pybel的分子形式，不然报错
            raise TypeError('molecule must be pybel.Molecule object,'
                            ' %s was given' % type(molecule))
        if molcode is None: # 确定molcode没问题，不然报错
            if self.save_molecule_codes is True:
                raise ValueError('save_molecule_codes is set to True,'
                                 ' you must specify code for the molecule')
        elif not isinstance(molcode, (float, int)):
            raise TypeError('motlype must be float, %s was given'
                            % type(molcode))

        coords = []
        features = []
        heavy_atoms = []

        for i, atom in enumerate(molecule): # 遍历molecule列表的原子
            # ignore hydrogens and dummy atoms (they have atomicnum set to 0)
            if atom.atomicnum > 1: # 确保原子不是杂原子或虚原子
                heavy_atoms.append(i) # 重原子的index
                coords.append(atom.coords) # 重原子的坐标

                features.append(np.concatenate((
                    self.encode_num(atom.atomicnum),
                    [atom.__getattribute__(prop) for prop in self.NAMED_PROPS],
                    [func(atom) for func in self.CALLABLES],
                ))) # encode_num函数，prop，func，拼在了一起

        coords = np.array(coords, dtype=np.float32)
        features = np.array(features, dtype=np.float32) # 坐标和特征数据类型的转化
        if self.save_molecule_codes:
            features = np.hstack((features,
                                  molcode * np.ones((len(features), 1)))) # 给特征加了一列，判断原子来源
        features = np.hstack([features,
                              self.find_smarts(molecule)[heavy_atoms]]) # 加smarts里定义的特征

        if np.isnan(features).any(): # 判断特征中是否有NaN，有的话报错
            raise RuntimeError('Got NaN when calculating features')

        return coords, features # 该函数返回坐标和特征

    def to_pickle(self, fname='featurizer.pkl'): # 特征可以保存成pkl
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
    def from_pickle(fname): # 特征也可以从pkl中load
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
