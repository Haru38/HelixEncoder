# -*- coding: utf-8 -*-
"""
@Time:Created on 2021/11/
@author: LiFan Chen
@Software: PyCharm

@Time:Modified on 2023/3/
@File modifier: Haruki Yamane
@Filename: mol_featurizer_forHelix.py
"""
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors
import pickle

num_atom_feat = 34


def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(
            x, allowable_set))
    return [x == s for s in allowable_set]


def one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return [x == s for s in allowable_set]


def atom_features(atom, explicit_H=False, use_chirality=True):
    """Generate atom features including atom symbol(10),degree(7),formal charge,
    radical electrons,hybridization(6),aromatic(1),Chirality(3)
    """
    symbol = ['C', 'N', 'O', 'F', 'P', 'S', 'Cl', 'Br', 'I', 'other']  # 10-dim
    degree = [0, 1, 2, 3, 4, 5, 6]  # 7-dim
    hybridizationType = [
        Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2, 'other'
    ]  # 6-dim
    results = one_of_k_encoding_unk(atom.GetSymbol(),symbol) + \
                  one_of_k_encoding(atom.GetDegree(),degree) + \
                  [atom.GetFormalCharge(), atom.GetNumRadicalElectrons()] + \
                  one_of_k_encoding_unk(atom.GetHybridization(), hybridizationType) + [atom.GetIsAromatic()]  # 10+7+2+6+1=26

    # In case of explicit hydrogen(QM8, QM9), avoid calling `GetTotalNumHs`
    if not explicit_H:
        results = results + one_of_k_encoding_unk(atom.GetTotalNumHs(),
                                                  [0, 1, 2, 3, 4])  # 26+5=31
    if use_chirality:
        try:
            results = results + one_of_k_encoding_unk(
                atom.GetProp('_CIPCode'),
                ['R', 'S']) + [atom.HasProp('_ChiralityPossible')]
        except:
            results = results + [False, False] + [
                atom.HasProp('_ChiralityPossible')
            ]  # 31+3 =34
    return results


def adjacent_matrix(mol):
    adjacency = Chem.GetAdjacencyMatrix(mol)
    return np.array(adjacency) + np.eye(adjacency.shape[0])


def mol_features(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
    except:
        raise RuntimeError("SMILES cannot been parsed!")
    #mol = Chem.AddHs(mol)
    atom_feat = np.zeros((mol.GetNumAtoms(), num_atom_feat))
    for atom in mol.GetAtoms():
        atom_feat[atom.GetIdx(), :] = atom_features(atom)
    adj_matrix = adjacent_matrix(mol)
    return atom_feat, adj_matrix


if __name__ == "__main__":

    from word2vec import seq_to_kmers, get_protein_embedding
    from gensim.models import Word2Vec
    import os

    DATASET = "test_ECL2"
    with open(
            "../data/data_split_TM_ECL/test_ECL2_0.txt",
            "r") as f:
        data_list = f.read().strip().split('\n')
    not_list_test = []
    for i, d in enumerate(data_list):
        if '.' in d.strip().split()[0]:
            not_list_test.append(i)

    data_list = [d for d in data_list if '.' not in d.strip().split()[0]]
    N = len(data_list)
    print(N)

    compounds, adjacencies, proteins, interactions = [], [], [], []
    model = Word2Vec.load("word2vec_30.model")

    for no, data in enumerate(data_list):
        print('/'.join(map(str, [no + 1, N])))
        split_data = data.strip().split(" ")
        smiles = split_data[0]
        #for TM
        # sequences = split_data[1:8]
        # interaction = split_data[8]
        #for TM + ECL
        sequences = split_data[1:9]
        interaction = split_data[9]

        atom_feature, adj = mol_features(smiles)
        compounds.append(atom_feature)
        adjacencies.append(adj)
        interactions.append(np.array([float(interaction)]))

        this_protein = []
        for sequence in sequences:
            protein_embedding = get_protein_embedding(model,
                                                      seq_to_kmers(sequence))
            this_protein.append(protein_embedding)
        proteins.append(this_protein)

    dir_input = ('../dataset/' + DATASET + '/word2vec_30/')
    os.makedirs(dir_input, exist_ok=True)
    np.save(dir_input + 'compounds', compounds)
    np.save(dir_input + 'adjacencies', adjacencies)
    np.save(dir_input + 'proteins', proteins)
    np.save(dir_input + 'interactions', interactions)
    print('The preprocess of ' + DATASET + ' dataset has finished!')