import torch
import os
from pyfasta import Fasta
import numpy as np
import pandas as pd
import time

acgt2num = {'A': 0,
            'C': 1,
            'G': 2,
            'T': 3}

def seq2mat(seq):
    seq = seq.upper()
    h = 4
    w = len(seq)
    mat = np.zeros((h, w), dtype=bool)  # True or false in mat
    for i in range(w):
        if seq[i] != 'N':
            mat[acgt2num[seq[i]], i] = 1.
    return mat

genome = Fasta('hg38.fa')

def model_predict(model,seq_embeds,tf_feature):
    seq_embeds = torch.from_numpy(seq_embeds)
    tf_feature = np.pad(tf_feature,((0, 0), (0, 1)),'constant',constant_values = (0,0))
    tf_feature = np.expand_dims(tf_feature,axis=0)
    tf_feature = torch.from_numpy(tf_feature)
    seq_embeds = seq_embeds.type(torch.FloatTensor)
    tf_feature = tf_feature.type(torch.FloatTensor)
    seq_embeds = seq_embeds.to('cuda')
    tf_feature = tf_feature.to('cuda')
    signals = model(seq_embeds,tf_feature)
    np_signals = signals.cpu().detach().numpy()
    return np_signals


def load_weights(model,path):
    model.load_state_dict(torch.load('pretrainModel/model.ckpt',map_location='cuda:0')['state_dict'])
    model = model.eval()
    model = model.cuda()
    return model

def get_motif_score(tf_list, motif2tf, isContain, region_file, motifscan_file, save = True):
    """parse the motifscan results by homer tool.
    Args:
        region_file: path to bed file that was used for motif scanning
        motifscan_file: path to the motifscan file from Homer tool
    """
    regions = ['_'.join(item.split('\t')[:4]) for item in open(region_file).readlines()]
    motifscore = np.empty((len(regions), len(tf_list)), dtype = 'float32')
    count = 0
    f_motifscan = open(motifscan_file,'r')
    #skip header
    line = f_motifscan.readline()
    print(line)
    line = f_motifscan.readline()
    print(line)
    while line != '':
        motif = line.split('\t')[3]
        if isContain[motif]:
            region_id = line.split('\t')[0]
            score = float(line.split('\t')[-1].rstrip())
            for tf in motif2tf[motif]:
                previous = motifscore[int(region_id)-1][tf_list.index(tf)]
                motifscore[int(region_id)-1][tf_list.index(tf)] = max(score, previous)
        line = f_motifscan.readline()
        count += 1
    f_motifscan.close()
    np.save('motifscore.npy', motifscore)
