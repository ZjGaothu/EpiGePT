import numpy as np
import os
import pandas as pd
import argparse
import torch
import argparse
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
torch.backends.cudnn.deterministic = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from model.config import *
from model import EpiGePT,dataset


def main(hparams):
    pl.seed_everything(hparams.seed)
    model = EpiGePT.EpiGePT_pretrain(WORD_NUM,SEQUENCE_DIM,TF_DIM,BATCH_SIZE)
    np.random.seed(123)
    test_idx = np.load(hparams.cell_idxs_path)
    model.load_state_dict(torch.load(hparams.pretrained_model_path,map_location='cuda:0')['state_dict'])
    model.eval()
    model.cuda()
    if hparams.pred_method == 0:
        """
        cross cell type prediction on the same genomic regions
        """
        for i in range(len(test_idx)):
            predicted_proba = []
            true_label = []
            print('cell index ' + str(test_idx[i]) )
            test_data_loader = torch.utils.data.DataLoader(dataset.GenomicData(np.array([test_idx[i]]),'data',hparams.num_train_region),batch_size=BATCH_SIZE, shuffle=False,num_workers=16)
            for inputs,tf_feats,labels in test_data_loader:
                pre_batch = model(inputs,tf_feats)
                pre_batch = pre_batch.cpu.detach().numpy()
                pre_batch = pre_batch.reshape((pre_batch.shape[0]*pre_batch.shape[1],NUM_SIGNALS))
                predicted_proba.extend(pre_batch.tolist())
                labels = labels.detach().numpy()
                labels = labels.reshape(-1,NUM_SIGNALS)
                true_label.extend(labels)
            predicted_proba = np.array(predicted_proba)
            predicted_proba = pd.DataFrame(predicted_proba)
            predicted_proba.to_csv('%s/predicted_proba_cell_type%s.csv'%(hparams.pred_path,i), index=None, header=None)
            np.save('%s/true_label_%s.npy'%(hparams.pred_path,i),true_label)
    elif hparams.pred_method == 1:
        """
        cross cell region prediction on the same genomic regions
        """
        train_idx = np.load(hparams.cell_idxs_path)
        for i in range(len(train_idx)):
            predicted_proba = []
            true_label = []
            print('cell index ' + str(train_idx[i]) )
            test_data_loader = torch.utils.data.DataLoader(dataset.GenomicData(np.array([train_idx[i]]),'data',hparams.num_train_region,isTrain=False),batch_size=BATCH_SIZE, shuffle=False,num_workers=16)
            for inputs,tf_feats,labels in test_data_loader:
                pre_batch = model(inputs,tf_feats)
                pre_batch = pre_batch.cpu.detach().numpy()
                pre_batch = pre_batch.reshape((pre_batch.shape[0]*pre_batch.shape[1],NUM_SIGNALS))
                predicted_proba.extend(pre_batch.tolist())
                labels = labels.detach().numpy()
                labels = labels.reshape(-1,NUM_SIGNALS)
                true_label.extend(labels)
            predicted_proba = np.array(predicted_proba)
            predicted_proba = pd.DataFrame(predicted_proba)
            predicted_proba.to_csv('%s/predicted_proba_cell_type%s.csv'%(hparams.pred_path,i), index=None, header=None)
            np.save('%s/true_label_%s.npy'%(hparams.pred_path,i),true_label)
    elif hparams.pred_method == 2:
        """
        cross cell both prediction on the same genomic regions
        """
        for i in range(len(test_idx)):
            predicted_proba = []
            true_label = []
            print('cell index ' + str(test_idx[i]) )
            test_data_loader = torch.utils.data.DataLoader(dataset.GenomicData(np.array([test_idx[i]]),'data',hparams.num_train_region,isTrain=False),batch_size=BATCH_SIZE, shuffle=False,num_workers=16)
            for inputs,tf_feats,labels in test_data_loader:
                pre_batch = model(inputs,tf_feats)
                pre_batch = pre_batch.cpu.detach().numpy()
                pre_batch = pre_batch.reshape((pre_batch.shape[0]*pre_batch.shape[1],NUM_SIGNALS))
                predicted_proba.extend(pre_batch.tolist())
                labels = labels.detach().numpy()
                labels = labels.reshape(-1,NUM_SIGNALS)
                true_label.extend(labels)
            predicted_proba = np.array(predicted_proba)
            predicted_proba = pd.DataFrame(predicted_proba)
            predicted_proba.to_csv('%s/predicted_proba_cell_type%s.csv'%(hparams.pred_path,i), index=None, header=None)
            np.save('%s/true_label_%s.npy'%(hparams.pred_path,i),true_label)
    else:
        return 0
            
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42, help="Reproduction")
    parser.add_argument('--gpus', type=int, default=1, help="How many gpus")
    parser.add_argument("--pred_method", type=int, default=0, help="Reproduction")
    parser.add_argument('--pretrain_model_path', type=str, default='./checkpoint/pretrain_model.ckpt',help="path of the pretrained-model")
    parser.add_argument('--save_model_path', type=str, default='checkpoint',help="path to save the model")
    parser.add_argument('--cell_idxs_path', type=str, default='test_cell_type_idxs.npy',help="path to the indexs of the subset of the cell types")
    parser.add_argument('--pred_path', type=str, default='checkpoint',help="path to the indexs of the subset of the cell types")
    parser.add_argument("--num_train_region", type=int, default=10000, help="number of training genomic regions")
    hparams = parser.parse_args()
    main(hparams)
