import torch
import os
from pyfasta import Fasta
import numpy as np
import pandas as pd

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