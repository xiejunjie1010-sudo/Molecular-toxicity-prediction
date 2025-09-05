from rdkit.Chem import MACCSkeys, AllChem
from rdkit import DataStructs, Chem
from rdkit.Chem.Draw import SimilarityMaps

# ms = [Chem.MolFromSmiles('C1CCC1OCC'), Chem.MolFromSmiles('CC(C)OCC'), Chem.MolFromSmiles('CCOCC')]
# # fps = [MACCSkeys.GenMACCSKeys(x) for x in ms]
#
# fp = SimilarityMaps.GetAPFingerprint(ms[0], fpType='bv')
# print(type(fp))
# print(fp.ToBitString())
# print(fp.GetNumBits())


smiles_one = 'C1CCC1OCC'
mol = Chem.MolFromSmiles(smiles_one,True)
for j in range(10):
    smi = Chem.MolToSmiles(mol, doRandom=True)
    print(smi)

# >>> from rdkit.Chem.Draw import SimilarityMaps
# >>> fp = SimilarityMaps.GetAPFingerprint(mol, fpType='normal')
# >>> fp1 = AllChem.GetAtomPairFingerprint(mol)
# >>> print(fp == fp1)
# True
# >>> fp = SimilarityMaps.GetTTFingerprint(mol, fpType='normal')
# >>> fp1 = AllChem.GetTopologicalTorsionFingerprint(mol)
# >>> print(fp == fp1)
# True
# >>> fp = SimilarityMaps.GetMorganFingerprint(mol, fpType='bv')
# >>> fp1 = AllChem.GetMorganFingerprintAsBitVect(mol, 2)
# >>> print(fp == fp1)
# True

# print(DataStructs.FingerprintSimilarity(fps[1], fps[1]))

# l1 = 0
# l2 = 0
# for data in datasets:
#     if data.y ==1:
#         l1 = l1+1
#     else:
#         l2 = l2+1
# print(l1,l2)
