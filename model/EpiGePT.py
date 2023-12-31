from transformers import BertConfig,BertModel
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import math
import argparse

import torch.nn.functional as F
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from dataset import GenomicData
from config import *

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.backends.cudnn.deterministic = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Convmodule(nn.Module):
    """Convolution Module.
    The convolution module is made up of "num_cb" conv+pooling blocks.
    """
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
    def forward(self,x):
        x = self.pool1(self.relu(self.conv1(x)))
        x = self.pool2(self.relu(self.conv2(x)))
        x = self.pool3(self.relu(self.conv3(x)))
        x = self.pool4(self.relu(self.conv4(x)))
        x = self.pool5(self.relu(self.conv5(x)))
        return x
    
class Multitaskmodule(nn.Module):
    """Multi-task prediction module.
    This module is mainly made up with linear layer.
    """
    def __init__(self,SEQUENCE_DIM,TF_DIM,NUM_SIGNALS):
        super().__init__()
        self.linear = nn.Linear(SEQUENCE_DIM + TF_DIM, NUM_SIGNALS)

    def forward(self,x):
        x = F.relu(self.linear(x))
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
    """Initialize layers to build EpiGePT model.
        Args:
            word_num: size of the vocabulary of the transformer module.
            sequence_dim: dimension of the token embedding from the output of the Convolution module.
            tf_dim: dimension of the TF embedding.
            batch_size: batch size for training.
    """
    def __init__(self, word_num,sequence_dim,tf_dim,batch_size):
        super().__init__()
        self.word_num = word_num
        self.sequence_dim = sequence_dim
        self.tf_dim = tf_dim
        self.batch_size = batch_size
        self.convmodule = Convmodule(CHANNEL_SIZE,SEQUENCE_DIM)
        self.config_encoder = BertConfig(vocab_size=word_num, hidden_size=SEQUENCE_DIM + TF_DIM,
                                            num_hidden_layers=NUM_LAYER,
                                            num_attention_heads=NUM_HEAD,
                                            intermediate_size=1024,
                                            output_hidden_states=False,
                                            output_attentions=False,
                                            max_position_embeddings=1000)#shape (bs, inp_len, inp_len)

        self.transformermodule = BertModel(config=self.config_encoder)
        # Linear layer for multi-task prediction
        self.multitaskmodule = Multitaskmodule(SEQUENCE_DIM,TF_DIM,NUM_SIGNALS)

    def forward(self,batch_inputs_seq,batch_inputs_tf):
        x = self.convmodule(batch_inputs_seq)
        x = x.transpose(1,2)
        x = torch.cat([x ,batch_inputs_tf],dim=2)
        x = self.transformermodule(inputs_embeds=x)
        output = self.multitaskmodule(x[0])
        return output
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(),lr = LEARNING_RATE)
    
    def training_step(self,batch,batch_idx):
        batch_encoder_embeds,batch_inputs_tf,targets = batch
        batch_pre = self.forward(batch_encoder_embeds,batch_inputs_tf)
        batch_pre = batch_pre.view(-1,NUM_SIGNALS)
        targets = targets.view(-1, NUM_SIGNALS)
        loss_fn = nn.MSELoss(reduce=True, size_average=True)
        loss = loss_fn(batch_pre,targets)
        return loss
    def validation_step(self,batch,batch_idx):
        batch_encoder_embeds, batch_inputs_tf,targets = batch
        batch_pre = self.forward(batch_encoder_embeds,batch_inputs_tf)
        batch_pre = batch_pre.view(-1, NUM_SIGNALS)
        loss_fn = nn.MSELoss(reduce=True, size_average=True)
        targets = targets.view(-1,NUM_SIGNALS)
        loss = loss_fn(batch_pre,targets)
        self.log('val_loss', loss)
        return loss

    def setup(self,stage):
        parser = argparse.ArgumentParser()
        parser.add_argument("--num_train_region", type=int, default=10000, help="number of training genomic regions")
        hparams = parser.parse_args()
        np.random.seed(123)
        train_idx = np.random.choice(np.arange(28),size=20,replace=False)
        test_idx = np.array([item for item in np.arange(28) if item not in train_idx])
        num_train_region = hparams.num_train_region
        dataset = GenomicData(train_idx,'data',num_train_region)
        train_size = int(0.9 * len(dataset))  
        np.random.seed(123)
        self.dataset_train, self.dataset_val = torch.utils.data.random_split(dataset,
                                            [train_size, len(dataset) - train_size])
        val_size = int(0.5*len(self.dataset_val))
        np.random.seed(123)
        self.dataset_val,self.dataset_test = torch.utils.data.random_split(self.dataset_val,
                                            [val_size, len(self.dataset_val) - val_size])
        self.dataset_test = GenomicData(test_idx,'data',num_train_region)

    def train_dataloader(self):
        return DataLoader(self.dataset_train, batch_size=self.batch_size, shuffle=True, drop_last=False,num_workers=16)

    def val_dataloader(self):
        return DataLoader(self.dataset_val, batch_size=self.batch_size, num_workers=16)

    def test_dataloader(self):
        return DataLoader(self.dataset_test, batch_size=self.batch_size) 
    
class EpiGePT_pretrain(pl.LightningModule):
    """Initialize layers to build EpiGePT_pretrain model.
        Args:
            word_num: size of the vocabulary of the transformer module.
            sequence_dim: dimension of the token embedding from the output of the Convolution module.
            tf_dim: dimension of the TF embedding.
            batch_size: batch size for training.
    """
    def __init__(self, word_num,sequence_dim,tf_dim,batch_size):
        super().__init__()
        self.word_num = word_num
        self.sequence_dim = sequence_dim
        self.tf_dim = tf_dim
        self.batch_size = batch_size
        self.convmodule = Convmodule(CHANNEL_SIZE,SEQUENCE_DIM)
        self.config_encoder = BertConfig(vocab_size=word_num, hidden_size=SEQUENCE_DIM + TF_DIM,
                                            num_hidden_layers=NUM_LAYER,
                                            num_attention_heads=NUM_HEAD,
                                            intermediate_size=1024,
                                            output_hidden_states=False,
                                            output_attentions=False,
                                            max_position_embeddings=1000)#shape (bs, inp_len, inp_len)

        self.encoder = BertModel(config=self.config_encoder)
        # Linear layer for multi-task prediction
        self.fc2 = nn.Linear(SEQUENCE_DIM + TF_DIM, NUM_SIGNALS)

    def forward(self,batch_inputs_seq,batch_inputs_tf):
        x = self.convmodule(batch_inputs_seq)
        x = x.transpose(1,2)
        x = torch.cat([x ,batch_inputs_tf],dim=2)
        x = self.encoder(inputs_embeds=x)
        output = F.relu(self.fc2(x[0]))
        return output
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(),lr = LEARNING_RATE)
    
    def training_step(self,batch,batch_idx):
        batch_encoder_embeds,batch_inputs_tf,targets = batch
        batch_pre = self.forward(batch_encoder_embeds,batch_inputs_tf)
        batch_pre = batch_pre.view(-1,NUM_SIGNALS)
        targets = targets.view(-1, NUM_SIGNALS)
        loss_fn = nn.MSELoss(reduce=True, size_average=True)
        loss = loss_fn(batch_pre,targets)
        return loss
    def validation_step(self,batch,batch_idx):
        batch_encoder_embeds, batch_inputs_tf,targets = batch
        batch_pre = self.forward(batch_encoder_embeds,batch_inputs_tf)
        batch_pre = batch_pre.view(-1, NUM_SIGNALS)
        loss_fn = nn.MSELoss(reduce=True, size_average=True)
        targets = targets.view(-1,NUM_SIGNALS)
        loss = loss_fn(batch_pre,targets)
        self.log('val_loss', loss)
        return loss
        


    def setup(self,stage):
        parser = argparse.ArgumentParser()
        parser.add_argument("--num_train_region", type=int, default=10000, help="number of training genomic regions")
        hparams = parser.parse_args()
        np.random.seed(123)
        train_idx = np.random.choice(np.arange(28),size=20,replace=False)
        test_idx = np.array([item for item in np.arange(28) if item not in train_idx])
        num_train_region = hparams.num_train_region
        dataset = GenomicData(train_idx,'data',num_train_region)
        train_size = int(0.9 * len(dataset))  
        np.random.seed(123)
        self.dataset_train, self.dataset_val = torch.utils.data.random_split(dataset,
                                            [train_size, len(dataset) - train_size])
        val_size = int(0.5*len(self.dataset_val))
        np.random.seed(123)
        self.dataset_val,self.dataset_test = torch.utils.data.random_split(self.dataset_val,
                                            [val_size, len(self.dataset_val) - val_size])
        self.dataset_test = GenomicData(test_idx,'data',num_train_region)

    def train_dataloader(self):
        return DataLoader(self.dataset_train, batch_size=self.batch_size, shuffle=True, drop_last=False,num_workers=16)

    def val_dataloader(self):
        return DataLoader(self.dataset_val, batch_size=self.batch_size, num_workers=16)

    def test_dataloader(self):
        return DataLoader(self.dataset_test, batch_size=self.batch_size) 
