# -*- coding: utf-8 -*-
"""
@Time: Created on 2023/3/
@File modifier: Haruki Yamane
@Filename: predict.py
@Software: PyCharm
"""

from helix_ecl2_model import *
import numpy as np
import random
import time
import timeit

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import math
import numpy as np
from sklearn.metrics import roc_auc_score, precision_score, recall_score, precision_recall_curve, auc


def load_tensor(file_name, dtype):
    return [dtype(d).to(device) for d in np.load(file_name + '.npy',allow_pickle=True)]

def load_tensor_for_TM(file_name, dtype):
    proteins = np.load(file_name + '.npy', allow_pickle=True)
    torch_tensor_protein = []
    for protein in proteins:
        this_protein = []
        for helix in protein:
            this_protein.append(dtype(helix).to(device))
        torch_tensor_protein.append(this_protein)
    return torch_tensor_protein

def open_file(name):
  with open(name,"r") as f:
    data_list = f.read().strip().split('\n')
  result = []
  for data in data_list:
    result.append(data.split("\t"))
  valAUC = [float(d[3]) for i,d in enumerate(result) if i != 0]
  trainLoss = [float(d[2]) for i,d in enumerate(result) if i != 0]
  return valAUC,trainLoss


if __name__ == "__main__":
  """CPU or GPU"""
  if torch.cuda.is_available():
      device = torch.device('cuda:0')
      print('The code uses GPU...')
  else:
      device = torch.device('cpu')
      print('The code uses CPU!!!')

  """ create model ,trainer and tester """
  protein_dim = 100
  atom_dim = 34
  hid_dim = 64
  n_layers = 3
  n_heads = 8
  pf_dim = 256
  dropout = 0.1
  kernel_size = 3
  encoder_n_heads = 8

  encoder = Encoder(protein_dim, hid_dim, n_layers, kernel_size, dropout,
                  device,encoder_n_heads, pf_dim,
                  EncoderLayer, SelfAttention, PositionwiseFeedforward)
  decoder = Decoder(atom_dim, hid_dim, n_layers, n_heads, pf_dim,
                        DecoderLayer, SelfAttention, PositionwiseFeedforward,
                        dropout, device)
  model = Predictor(encoder, decoder, device)

  path = "output/model/"
  model_name = "helixEncoder_TM_ECL2"
  DATASET = "test_ECL2"  #./dataset/test_ECL2

  model.to(device)
  model.load_state_dict(torch.load(path + model_name,map_location = device))

  dir_input = ('dataset/' + DATASET + '/word2vec_30/')
  compounds = load_tensor(dir_input + 'compounds', torch.FloatTensor)
  adjacencies = load_tensor(dir_input + 'adjacencies', torch.FloatTensor)
  proteins = load_tensor_for_TM(dir_input + 'proteins', torch.FloatTensor)
  interactions = load_tensor(dir_input + 'interactions', torch.LongTensor)
  dataset = list(zip(compounds, adjacencies, proteins, interactions))
  
  with torch.no_grad():
    model.eval()
    correct_labels, predicted_labels, predicted_scores,encoder_attention,decoder_attention = model(dataset, train=False)

  print("model_name : ",model_name)
  print(f"collect labels:")
  print(correct_labels)
  print(f"predict labels: ")
  print(predicted_labels)
  print(f"predict scores: ")
  print(predicted_scores)

  tester = Tester(model)
  AUC, PRC = tester.test(dataset)
  print(f"test : {AUC}")
  print(f"test PRC : {PRC}")

