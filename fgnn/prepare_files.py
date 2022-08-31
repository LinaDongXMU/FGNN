import os

def pocket_mol2(path):
     for filename in os.listdir(path):
         pocket_pdb=os.path.join(path,filename,str(filename)+'_pocket.pdb')
         pocket_mol2=os.path.join(path,filename,str(filename)+'_pocket.mol2')
         cmd='obabel -ipdb '+str(pocket_pdb)+' -omol2 -O '+str(pocket_mol2)
         os.system(cmd)
path1='./pdbbind2016/refined-set'
path2='./pdbbind2016/coreset'
pocket_mol2(path1)
pocket_mol2(path2)

