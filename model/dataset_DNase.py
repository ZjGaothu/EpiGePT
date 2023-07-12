import numpy as np
import torch
torch.set_default_tensor_type(torch.FloatTensor)
from torch.utils.data import Dataset
from pyfasta import Fasta
from config_DNase import *
import pandas as pd

acgt2num = {'A': 0,
            'C': 1,
            'G': 2,
            'T': 3}

class GenomicData(Dataset):
    def __init__(self, train_idx, path,quantile_norm=False):
        self.path = path
        self.genome = Fasta('%s/genome.fa'%path)
        self.train_idx = train_idx
        pd_openness = pd.read_csv('%s/readscount_normalized_filtered.csv'%path,header=0,index_col=[0],sep='\t')
        self.pd_openness = np.log(pd_openness+1)
        pd_tf_gexp = pd.read_csv('%s/tf_gexp.csv'%path ,sep='\t',header=0,index_col=[0])
        if quantile_norm:
            pd_tf_gexp = pd.DataFrame.transpose(self.quantile_norm_trans(pd.DataFrame.transpose(pd_tf_gexp)))
        self.pd_tf_gexp = np.log(pd_tf_gexp+1)
        self.pd_tf_bs = pd.read_csv('%s/tf_motif_score.csv'%path,sep='\t',header=0,index_col=[0])
        self.train_dseq_celllines = np.array(list(self.pd_openness.columns))[self.train_idx]
        self.train_rseq_celllines = np.array(list(self.pd_tf_gexp.index))[self.train_idx]

    def quantile_norm_trans(self, matrix):
        rank_mean = matrix.stack().groupby(matrix.rank(method='first').stack().astype(int)).mean()
        return matrix.rank(method='min').stack().astype(int).map(rank_mean).unstack()
    
    def seq2mat(self,seq):
        seq = seq.upper()
        h = 4
        w = len(seq)
        mat = np.zeros((h, w), dtype=bool)
        for i in range(w):
            if seq[i] != 'N':
                mat[acgt2num[seq[i]], i] = 1.
        return mat


    def get_seq_meta(self,file_path):
        pd_openness = pd.read_csv(file_path,header=0,index_col=[0],sep='\t',dtype='float16')
        seq_info, cellline_info = pd_openness.index, pd_openness.columns
        openness_mat = pd_openness.values
        return openness_mat,seq_info,cellline_info


    def get_seq_from_meta(self,region_idx):
        seq_info = self.pd_openness.index[region_idx*INPUT_LEN:(region_idx+1)*INPUT_LEN]
        seq_list = []
        for info in seq_info:
            chrom, start, end = info.split(':')[0],int(info.split(':')[1].split('-')[0]),int(info.split(':')[1].split('-')[1])
            seq_list.append(self.genome[chrom][start:end])
        return seq_list
        

    def get_seq_embeds(self, seq, k=4):
        dic = {'a':0, 'A':0, 'c':1, 'C':1, 'g':2, 'G':2, 't':3, 'T':3, 'N':4, 'n':4}
        assert len(seq) > k
        kmer_feat = np.zeros((4**k,),dtype='float16')
        for i in range(len(seq)-k+1):
            sub_seq = seq[i:(i+k)]
            if 'N' not in sub_seq and 'n' not in sub_seq:
                idx = sum([dic[char]*4**(k-i-1) for i,char in enumerate(sub_seq)])
                kmer_feat[idx]+=1
        return kmer_feat
    
    def get_tf_state(self,region_idx, cellline_idx):
        tf_gexp_vec = self.pd_tf_gexp.loc[self.train_rseq_celllines[cellline_idx]].values #(711,)
        tf_gexp_feat = np.tile(tf_gexp_vec,(INPUT_LEN,1))
        seq_info = self.pd_openness.index[region_idx*INPUT_LEN:(region_idx+1)*INPUT_LEN]
        seq_extend_info=[]
        for info in seq_info:
            chrom, start, end = info.split(':')[0],int(info.split(':')[1].split('-')[0]),int(info.split(':')[1].split('-')[1])
            seq_extend_info.append('%s:%d-%d'%(chrom,start-400,end+400))
        tf_bs_feat = self.pd_tf_bs.loc[seq_extend_info].values 
        return tf_gexp_feat*tf_bs_feat


    def __getitem__(self, index):
        region_idx = index // len(self.train_dseq_celllines)
        cellline_idx = index % len(self.train_dseq_celllines)  #0-99 bug here!
        seq_list = self.get_seq_from_meta(region_idx)
        assert len(seq_list) == INPUT_LEN
        seq_embeds_list = map(self.seq2mat,seq_list)
        seq_embeds = np.hstack(seq_embeds_list)
        tf_feats = self.get_tf_state(region_idx,cellline_idx)
        targets_openness = self.pd_openness[self.train_dseq_celllines[cellline_idx]].values[region_idx*INPUT_LEN:(region_idx+1)*INPUT_LEN]
        inputs_embeds = np.array(seq_embeds,dtype='float16')
        tf_feats = np.pad(tf_feats,((0, 0), (0, 1)),'constant',constant_values = (0,0))
        tf_feats = np.array(tf_feats,dtype = 'float16')
        targets_openness = np.array(targets_openness,dtype='float16')
        targets_openness = targets_openness[:,np.newaxis] 
        inputs_embeds = torch.from_numpy(inputs_embeds)
        tf_feats = torch.from_numpy(tf_feats)
        targets_openness = torch.from_numpy(targets_openness)
        inputs_embeds = inputs_embeds.type(torch.FloatTensor)
        tf_feats = tf_feats.type(torch.FloatTensor)
        targets_openness = targets_openness.type(torch.FloatTensor)
        
        return (inputs_embeds, tf_feats,targets_openness)

    def __len__(self):
        return len(self.train_dseq_celllines)*(self.pd_openness.shape[0] // INPUT_LEN)