# -*- coding: utf-8 -*-
"""
@Time:Created on 2019/9/17 8:54
@author: LiFan Chen
@Filename: main.py
@Software: PyCharm
"""
import torch
import numpy as np
import random
import os
import time
from model import *
import timeit
import warnings
warnings.simplefilter('ignore')


def load_tensor2(file_name, dtype):
    proteins = np.load(file_name + '.npy', allow_pickle=True)
    torch_tensor_protein = []
    for protein in proteins:
        this_protein = []
        for helix in protein:
            this_protein.append(dtype(helix).to(device))
        torch_tensor_protein.append(this_protein)
    return torch_tensor_protein


def load_tensor(file_name, dtype):
    return [
        dtype(d).to(device)
        for d in np.load(file_name + '.npy', allow_pickle=True)
    ]


def shuffle_dataset(dataset, seed):
    np.random.seed(seed)
    np.random.shuffle(dataset)
    return dataset


def split_dataset(dataset, ratio):
    n = int(ratio * len(dataset))
    dataset_1, dataset_2 = dataset[:n], dataset[n:]
    return dataset_1, dataset_2


if __name__ == "__main__":
    SEED = 1
    random.seed(SEED)
    torch.manual_seed(SEED)
    # torch.backends.cudnn.deterministic = True
    DATASET = "classA_position_reverse_train"
    print(DATASET)
    """CPU or GPU"""
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        print('The code uses GPU...')
    else:
        device = torch.device('cpu')
        print('The code uses CPU!!!')
    """Load preprocessed data."""
    dir_input = ('dataset/' + DATASET + '/word2vec_30/')
    compounds = load_tensor(dir_input + 'compounds', torch.FloatTensor)
    adjacencies = load_tensor(dir_input + 'adjacencies', torch.FloatTensor)
    proteins = load_tensor2(dir_input + 'proteins', torch.FloatTensor)
    interactions = load_tensor(dir_input + 'interactions', torch.LongTensor)
    """Create a dataset and split it into train/dev/test."""
    dataset = list(zip(compounds, adjacencies, proteins, interactions))
    dataset = shuffle_dataset(dataset, 1234)
    dataset_train, dataset_dev = split_dataset(dataset, 0.8)
    """ create model ,trainer and tester """
    protein_dim = 100
    atom_dim = 34
    #########################
    #hid_dim_enconder = 8
    #hid_dim_decoder = 56
    hid_dim = 64
    ########################
    n_layers = 3
    n_heads = 8
    pf_dim = 256
    dropout = 0.1
    batch = 64
    lr = 1e-4
    weight_decay = 1e-4
    decay_interval = 5
    lr_decay = 1.0
    iteration = 100
    kernel_size = 7
    encoder = Encoder(protein_dim, hid_dim, n_layers, kernel_size, dropout,
                      device)
    decoder = Decoder(atom_dim, hid_dim, n_layers, n_heads, pf_dim,
                      DecoderLayer, SelfAttention, PositionwiseFeedforward,
                      dropout, device)
    model = Predictor(encoder, decoder, device)
    # model.load_state_dict(torch.load("output/model/lr=0.001,dropout=0.1,lr_decay=0.5"))
    print(model)
    model.to(device)
    trainer = Trainer(model, lr, weight_decay, batch)
    tester = Tester(model)
    """Output files."""
    file_AUCs = 'output/result/classA_position_reverse_layerNorm' + '.txt'
    file_model = 'output/model/' + 'classA_position_reverse_layerNorm'
    AUCs = ('Epoch\tTime(sec)\tLoss_train\tAUC_dev\tPRC_dev')
    with open(file_AUCs, 'w') as f:
        f.write(AUCs + '\n')
    """Start training."""
    print('Training...')
    print(AUCs)
    start = timeit.default_timer()

    max_AUC_dev = 0
    for epoch in range(1, iteration + 1):
        #print(">>>"+str(epoch))
        if epoch % decay_interval == 0:
            trainer.optimizer.param_groups[0]['lr'] *= lr_decay

        loss_train = trainer.train(dataset_train, device)
        AUC_dev, PRC_dev = tester.test(dataset_dev)

        end = timeit.default_timer()
        time = end - start

        AUCs = [epoch, time, loss_train, AUC_dev, PRC_dev]
        tester.save_AUCs(AUCs, file_AUCs)
        if AUC_dev > max_AUC_dev:
            tester.save_model(model, file_model)
            max_AUC_dev = AUC_dev
        print('\t'.join(map(str, AUCs)))
