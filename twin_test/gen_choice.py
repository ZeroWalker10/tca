import sys
sys.path.append('..')

import os
import numpy as np
import json
import torch
from torch import nn
import torch.nn.functional as F
import model_zoo, dataset_zoo
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torchattacks.tools.util_display
from importlib import reload
import copy

import torchattacks.tools.transform_selection
from torchattacks.tools.transform_selection import TransformSelection
import model_zoo
import time
import pickle

from itertools import permutations
from tqdm import tqdm

def sample_k(imgs, slabels, tlabels, k=10):
    indexes = torch.arange(len(imgs))
    k_indexes = indexes[:k]
    return imgs[k_indexes], slabels[k_indexes], tlabels[k_indexes]

def can_squeeze(nodes, cur_node, cur_node_score):
    for d, dnodes in nodes.items():
        for tran_seq, score in dnodes.items():
            tran_seq_len = len(tran_seq[1:])
            cur_node_len = len(cur_node[1:])
            if tran_seq_len < cur_node_len:
                is_sub = any(tran_seq[1:] == cur_node[i:i+tran_seq_len]
                            for i in range(1, len(cur_node)-len(tran_seq)+2))
                if is_sub and score > cur_node_score: # sub translations
                    print('find sub trans:', tran_seq)
                    return True
    return False

datapath = '/home/zero/zero/split_dp/dataset/imagenet/new_adv_1k'
attack_book = './new_attack_book_1k.json'
mzoo = model_zoo.ModelZoo()
dzoo = dataset_zoo.DatasetZoo()
dname = 'imagenet'

with open(attack_book, 'r') as fp:
        attack_targets = json.load(fp)
        dataset = dzoo.load_dataset(dname, datapath)

select_n = np.inf 
imgs, slabels, tlabels = [], [], []
for i, ((img, label), (fname, _)) in enumerate(zip(dataset, dataset.imgs)):
    fname_basename = os.path.basename(fname)
    (_, tgt_label) = attack_targets[fname_basename]
    tgt_label = torch.LongTensor([tgt_label])
    src_label = torch.LongTensor([label])
    img = img.unsqueeze(0)
    imgs.append(img)
    slabels.append(src_label)
    tlabels.append(tgt_label)
    
    if i >= select_n:
        break
imgs = torch.cat(imgs)
slabels = torch.cat(slabels)
tlabels = torch.cat(tlabels)

tm = TransformSelection(p=1.0)
check_model = mzoo.pick_model('resnet34', dataset=dname).cuda()

k = 10
n_epoch = 10 
ngrp = 10 
times = 3
prev_nodes = {(-1,): 0.0}
tnodes = tuple(range(len(tm.aug_library['tran'])))
imgs = imgs.cuda()
slabels = slabels.cuda()
tlabels = tlabels.cuda()
nodes = {}

for mname in ['resnet50']:
    surrogate_model = mzoo.pick_model(mname, dataset=dname).cuda()
   
    for k in [10]:
        for depth in [2, 3, 4, 5]:
            print('test model:', mname, 'depth:', depth)
            nodes = {}
            start = time.time()
            for tran_seq in tqdm(permutations(range(len(tnodes)), depth)):
                scores = []
                for _ in range(times):
                    sub_imgs, sub_slabels, sub_tlabels = sample_k(imgs, slabels, tlabels, k=k)
                    score = tm.translation_score(surrogate_model, check_model, sub_imgs, sub_tlabels,
                                                n_epoch, tran_seq, ngrp)
                    scores.append(score)
                new_score = np.mean(scores)
                nodes[tran_seq] = new_score
            end = time.time()
            print('elapsed:', end - start)
            
            save_nodes = {depth: nodes}
            with open('new-{}-res34-{}-{}-{}.pickle'.format(mname, depth, k, dname), 'wb') as fp:
                pickle.dump(save_nodes, fp)
