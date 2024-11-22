from transformers import EncoderDecoderModel, BertConfig,EncoderDecoderConfig,BertTokenizer,BertModel
import numpy as np
import sys,os
import pandas as pd
import argparse
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score,roc_curve
import math


import torch.nn.functional as F
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from transformers import AdamW
from scipy.stats import pearsonr
torch.backends.cudnn.deterministic = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from model_hg38.config import *
from dataset3d import GenomicData

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class convmodule(nn.Module):
    def __init__(self,in_channels,out_channels,stride = 1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels = in_channels,out_channels = 32,kernel_size = 3,padding = 1 , stride = stride)
        self.pool1 = nn.MaxPool1d(kernel_size = 4)

        self.conv2 = nn.Conv1d(in_channels = 32 ,out_channels = 64,kernel_size = 5,padding = 2 , stride = stride)
        self.pool2 = nn.MaxPool1d(kernel_size = 4)

        self.conv3 = nn.Conv1d(in_channels = 64 ,out_channels = 96,kernel_size = 5,padding = 2 , stride = stride)
        self.pool3 = nn.MaxPool1d(kernel_size = 2)

        self.conv4 = nn.Conv1d(in_channels = 96,out_channels = 128,kernel_size = 3,padding = 1 , stride = stride) 
        self.pool4 = nn.MaxPool1d(kernel_size = 2)

        self.conv5 = nn.Conv1d(in_channels = 128 ,out_channels = out_channels,kernel_size = 3,padding = 1 , stride = stride)
        self.pool5 = nn.MaxPool1d(kernel_size = 2)
        self.relu=nn.ReLU()
        self.dropout = nn.Dropout(p=0.2)
    def forward(self,x):
        x = self.pool1(self.relu(self.conv1(x)))
        x = self.pool2(self.relu(self.conv2(x)))
        x = self.pool3(self.relu(self.conv3(x)))
        x = self.dropout(x)
        x = self.pool4(self.relu(self.conv4(x)))
        x = self.pool5(self.relu(self.conv5(x)))
        x = self.dropout(x)
        return x

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class EpiGePT(pl.LightningModule):
    def __init__(self, word_num, embedding_dim,batch_size):
        super().__init__()
        self.word_num = word_num
        self.embedding_dim = embedding_dim
        self.batch_size = batch_size

        self.convmodule = convmodule(4,256)


        self.config_encoder = BertConfig(vocab_size=word_num, hidden_size=256+712,
                                            num_hidden_layers=12,
                                            num_attention_heads=8,
                                            intermediate_size=1024,
                                            output_hidden_states=False,
                                            output_attentions=False,
                                            max_position_embeddings=1000)#shape (bs, inp_len, inp_len)

#         self.config = EncoderDecoderConfig.from_encoder_decoder_configs(self.config_encoder, self.config_decoder)
        self.encoder = BertModel(config=self.config_encoder)
        self.fc1 = nn.Linear(256+712, 128)
        self.fc2 = nn.Linear(128, 8)

    def forward(self,batch_inputs_seq,batch_inputs_tf):
        x = self.convmodule(batch_inputs_seq)
        x = x.transpose(1,2)
        x = torch.cat([x ,batch_inputs_tf],dim=2)
        x = self.encoder(inputs_embeds=x)
        x = F.relu(self.fc1(x[0]))
        x = self.fc2(x)
        output = F.relu(x)
        return output
    
    def configure_optimizers(self):
#         return AdamW(self.parameters(), lr=LEARNING_RATE)
        return torch.optim.Adam(self.parameters(),lr=LEARNING_RATE)
    
    def training_step(self,batch,batch_idx):
        batch_encoder_embeds,batch_inputs_tf,targets,targets_mask,targets_attn,attn_mask = batch
        batch_pre,attention_pre = self.forward(batch_encoder_embeds,batch_inputs_tf)
        batch_pre = batch_pre.view(-1,8)
        targets = targets.view(-1, 8)
        targets_mask = targets_mask.view(-1, 8)
        batch_pre = batch_pre * targets_mask
        loss_fn = nn.MSELoss(reduction='sum')
        loss = loss_fn(batch_pre,targets)
        num_unmasked_elements = torch.sum(targets_mask)
        loss = loss / (num_unmasked_elements+1e-8)
        attention_similarity = F.cosine_similarity(attention_pre*attn_mask, targets_attn)
        cosine_loss = -attention_similarity.mean()
        loss = loss + 2*cosine_loss
        return loss
    def validation_step(self,batch,batch_idx):
        batch_encoder_embeds, batch_inputs_tf,targets,targets_mask,targets_attn,attn_mask = batch
        batch_pre,attention_pre = self.forward(batch_encoder_embeds,batch_inputs_tf)
        batch_pre = batch_pre.view(-1, 8)
        targets_mask = targets_mask.view(-1, 8)
        batch_pre = batch_pre * targets_mask
        loss_fn = nn.MSELoss(reduction='sum')
        targets = targets.view(-1,8)
        loss = loss_fn(batch_pre,targets)
        num_unmasked_elements = torch.sum(targets_mask)
        loss = loss / (num_unmasked_elements+1e-8)
        attention_similarity = F.cosine_similarity(attention_pre*attn_mask, targets_attn)
        cosine_loss = -attention_similarity.mean()
        loss = loss + 2*cosine_loss
        self.log('val_loss', loss)
        


    def setup(self,stage):
        np.random.seed(123)
        train_idx = np.load('../train_cells.npy')
        test_idx = np.load('../test_cells.npy')
        dataset = GenomicData(train_idx)
        train_size = int(0.9 * len(dataset))  
        np.random.seed(123)
        self.dataset_train, self.dataset_val = torch.utils.data.random_split(dataset,
                                            [train_size, len(dataset) - train_size])


    def train_dataloader(self):
        return DataLoader(self.dataset_train, batch_size=self.batch_size, shuffle=True, drop_last=False,num_workers=10)

    def val_dataloader(self):
        return DataLoader(self.dataset_val, batch_size=self.batch_size, num_workers=10)