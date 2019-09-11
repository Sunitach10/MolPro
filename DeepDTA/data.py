
#################new dataloader
import torch
import json
import numpy as np
import pickle
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data.sampler import SubsetRandomSampler
import random
ligand_path = "kiba/ligands_can.txt"
protein_path = "kiba/proteins.txt"
affinity_path = "kiba/Y"



def MPLpair(mol, pro, y):
    mol_pro = []
    for i, m in mol.items():
        for j, p in pro.items():
            mol_pro.append(((torch.Tensor(m),torch.Tensor(p)),y[i][j]))
             #mol_pro.append(((m,p),y[i][j]))

    return mol_pro
# def MPy(l,p,y):
#     mpy=[]
#     for l,m in enumerate(l[:100]):
#         for k,p in enumerate(p[:100]):
#             mpy.append(((m,p),y[l][k]))
#     return mpy


def filter1(x, l):
    pt = {}
    for i, p in enumerate(x.values()):
        if len(p) <= l:
            pt[i] = p
    return pt


CHARPROT = {"A": 1, "C": 2, "B": 3, "E": 4, "D": 5, "G": 6,
            "F": 7, "I": 8, "H": 9, "K": 10, "M": 11, "L": 12,
            "O": 13, "N": 14, "Q": 15, "P": 16, "S": 17, "R": 18,
            "U": 19, "T": 20, "W": 21,
            "V": 22, "Y": 23, "X": 24,
            "Z": 25}

lenCHARPROT = 25

CHARmol = {"#": 1, "%": 2, ")": 3, "(": 4, "+": 5, "-": 6,
           ".": 7, "1": 8, "0": 9, "3": 10, "2": 11, "5": 12,
           "4": 13, "7": 14, "6": 15, "9": 16, "8": 17, "=": 18,
           "A": 19, "C": 20, "B": 21, "E": 22, "D": 23, "G": 24,
           "F": 25, "I": 26, "H": 27, "K": 28, "M": 29, "L": 30,
           "O": 31, "N": 32, "P": 33, "S": 34, "R": 35, "U": 36,
           "T": 37, "W": 38, "V": 39, "Y": 40, "[": 41, "Z": 42,
           "]": 43, "_": 44, "a": 45, "c": 46, "b": 47, "e": 48,
           "d": 49, "g": 50, "f": 51, "i": 52, "h": 53, "m": 54,
           "l": 55, "o": 56, "n": 57, "s": 58, "r": 59, "u": 60,
           "t": 61, "y": 62}

lenCHARmol = 62



def onehot(sent, charset, lens_m):
    # sent=list(sent.values())
    # sent=sent[:100]
    # sen_chars=set()
    # lens_m=[len(s) for s in sent.items()]
    # for s in sent.items():
    # sen_chars=sen_chars.union(set(s))

    onex_d = {}
    char_to_int = dict((c, i) for i, c in enumerate(list(charset.keys())))
    int_to_char = dict((i, c) for i, c in enumerate(list(charset.keys())))

    for i, s in sent.items():
        onehot_sent = np.zeros((len(char_to_int), lens_m))
        for j, char in enumerate(s):
            onehot_sent[char_to_int[char], j] = 1.0
        onex_d[i] = onehot_sent
    return onex_d
#custom data set
class NumbersDataset(Dataset):
    def __init__(self, ligand_path, protein_path, affinity_path):
        with open(ligand_path) as ligand_data:
            self.ligands = onehot(filter1(json.load(ligand_data), 50), CHARmol, 50)

            # self.ligands=self.ligands.values()
        with open(protein_path) as protein_data:
            self.proteins = onehot(filter1(json.load(protein_data), 600), CHARPROT, 600)

            # self.proteins=self.proteins.values()
        with open(affinity_path, 'rb') as Y:
            y1 = pickle.load(Y, encoding='latin1')
        self.y = torch.Tensor(np.nan_to_num(y1))

        self.mol_pro = MPLpair(self.ligands, self.proteins, self.y)

    def __len__(self):
        # return len(self.samples)

        return len(self.mol_pro)

    def __getitem__(self, idx):
        return self.mol_pro[idx]


# if __name__ == '__main__':
#     dataset = NumbersDataset(ligand_path, protein_path, affinity_path)
#     print(len(dataset))
#     #print(dataset[110])
#     # print(dataset[122:361])

#if __name__ == '__main__':
from torch.utils.data import DataLoader
#dataset = NumbersDataset(ligand_path, protein_path, affinity_path)
#dataloader = DataLoader(dataset, batch_size=10, shuffle=True)


################################################################train test set
def load_splitset(dataset,test_split):
    #test_split = .2
    shuffle_dataset = True
    random_seed= 42
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(test_split * dataset_size))
    if shuffle_dataset :
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    train_indices, test_indices = indices[split:], indices[:split]

    # Creating  data samplers and loaders:
    train_sampler = SubsetRandomSampler(train_indices)
    test_sampler = SubsetRandomSampler(test_indices)

    train_loader = torch.utils.data.DataLoader(dataset, batch_size=4,
                                           sampler=train_sampler)
    # train_loader =torch.Tensor(train_loader)

    test_loader = torch.utils.data.DataLoader(dataset, batch_size=1,
                                                sampler=test_sampler)
    # test_loader =torch.Tensor(test_loader)
    return train_loader,test_loader
# train_loader, test_loader = load_splitset(dataset, .2)
# print(len(train_loader))
# print(len(test_loader))
# print(type(train_loader))
