#!/usr/bin/env python
# coding:utf-8
"""
Author  : Xiwen Lu
Time    : 2022/3/24 8:54
Desc    : 
"""
import math
import os

import torch
import numpy as np
import torch.nn as nn
import yaml
from torch import optim
# from early_stoping import EarlyStopping
from dataset import *
from tqdm import trange


device = "cuda" if torch.cuda.is_available() else "cpu"
# torch.cuda.set_device(1)
# device = "cpu"

# MCformer Parameters

embeddings = []

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=10000):
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
        """
        :param x: [seq_len, batch_size, d_model]
        """
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


def get_attn_pad_mask(seq_q, seq_k):
    """
    :param seq_q: [batch_size, seq_len]
    :param seq_k: [batch_size, seq_len]
    :param seq_len could be src_len or it could be tgt_len
    :param seq_len in seq_q and seq_len in seq_k maybe not equal
    """
    batch_size, len_q = seq_q.size()
    batch_size, len_k = seq_k.size()
    # eq(zero) is PAD token
    # ! for Molecular Communication, here eq(0) as pad is wrong. need to be edit. it means we don't use the padding
    pad_attn_mask = seq_k.data.eq(-1).unsqueeze(1)  # [batch_size, 1, len_k], False is masked
    return pad_attn_mask.expand(batch_size, len_q, len_k)  # [batch_size, len_q, len_k]


# %%

class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V, d_k, attn_mask):
        """
        :param Q: [batch_size, n_heads, len_q, d_k]
        :param K: [batch_size, n_heads, len_k, d_k]
        :param V: [batch_size, n_heads, len_v(=len_k), d_v]
        :param d_k: the dimension of key
        :param attn_mask: [batch_size, n_heads, seq_len, seq_len]
        """
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k)  # scores : [batch_size, n_heads, len_q, len_k]
        scores.masked_fill_(attn_mask, -1e9)  # Fills elements of self tensor with value where mask is True.

        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)  # [batch_size, n_heads, len_q, d_v]
        return context, attn


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, d_k, d_v):
        super(MultiHeadAttention, self).__init__()
        self.n_heads = n_heads
        self.d_k = d_k
        self.d_v = d_v
        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias=False)
        self.fc = nn.Linear(n_heads * d_v, d_model, bias=False)
        self.LayerNorm = nn.LayerNorm(d_model)

    def forward(self, input_Q, input_K, input_V, attn_mask):
        '''
        :param input_Q: [batch_size, len_q, d_model]
        :param input_K: [batch_size, len_k, d_model]
        :param input_V: [batch_size, len_v(=len_k), d_model]
        :param attn_mask: [batch_size, seq_len, seq_len]
        '''
        residual, batch_size = input_Q, input_Q.size(0)
        # (B, S, D) -proj-> (B, S, D_new) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        Q = self.W_Q(input_Q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)  # Q: [batch_size, n_heads, len_q, d_k]
        K = self.W_K(input_K).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)  # K: [batch_size, n_heads, len_k, d_k]
        V = self.W_V(input_V).view(batch_size, -1, self.n_heads, self.d_v).transpose(1,
                                                                       2)  # V: [batch_size, n_heads, len_v(=len_k), d_v]

        attn_mask = attn_mask.unsqueeze(1).repeat(1, self.n_heads, 1, 1)  # attn_mask : [batch_size, n_heads, seq_len, seq_len]

        # context: [batch_size, n_heads, len_q, d_v], attn: [batch_size, n_heads, len_q, len_k]
        context, attn = ScaledDotProductAttention()(Q, K, V, d_k=self.d_k, attn_mask=attn_mask)
        context = context.transpose(1, 2).reshape(batch_size, -1,
                                                  self.n_heads * self.d_v)  # context: [batch_size, len_q, n_heads * d_v]
        output = self.fc(context)  # [batch_size, len_q, d_model]
        return self.LayerNorm(output + residual), attn


class PoswiseFeedForwardNet(nn.Module):
    def __init__(self, d_model, d_feedforward):
        super(PoswiseFeedForwardNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_feedforward, bias=False),
            nn.ReLU(),
            nn.Linear(d_feedforward, d_model, bias=False)
        )
        self.LayerNorm = nn.LayerNorm(d_model)

    def forward(self, inputs):
        '''
        :param inputs: [batch_size, seq_len, d_model]
        '''
        residual = inputs
        output = self.fc(inputs)
        return self.LayerNorm(output + residual)  # [batch_size, seq_len, d_model]


class EncoderLayer(nn.Module):
    def __init__(self, d_model, d_feedforward, n_heads, d_k, d_v):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention(d_model=d_model, n_heads=n_heads,d_k=d_k,d_v=d_v)
        self.pos_ffn = PoswiseFeedForwardNet(d_model=d_model,d_feedforward=d_feedforward)

    def forward(self, enc_inputs, enc_self_attn_mask):
        '''
        :param enc_inputs: [batch_size, src_len, d_model]
        :param enc_self_attn_mask: [batch_size, src_len, src_len]
        '''
        # enc_outputs: [batch_size, src_len, d_model], attn: [batch_size, n_heads, src_len, src_len]
        enc_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs,
                                               enc_self_attn_mask)  # enc_inputs to same Q,K,V
        enc_outputs = self.pos_ffn(enc_outputs)  # enc_outputs: [batch_size, src_len, d_model]
        return enc_outputs, attn


class DecoderLayer(nn.Module):
    def __init__(self, d_model, d_feedforward, n_heads,d_k,d_v):
        super(DecoderLayer, self).__init__()
        self.dec_self_attn = MultiHeadAttention(d_model=d_model, n_heads=n_heads,d_k=d_k,d_v=d_v)
        self.dec_enc_attn = MultiHeadAttention(d_model=d_model, n_heads=n_heads,d_k=d_k,d_v=d_v)
        self.pos_ffn = PoswiseFeedForwardNet(d_model=d_model, d_feedforward=d_feedforward)

    def forward(self, dec_inputs, enc_outputs, dec_self_attn_mask, dec_enc_attn_mask):
        """
        :param dec_inputs: [batch_size, tgt_len, d_model]
        :param enc_outputs: [batch_size, src_len, d_model]
        :param dec_self_attn_mask: [batch_size, tgt_len, tgt_len]
        :param dec_enc_attn_mask: [batch_size, tgt_len, src_len]
        """
        # dec_outputs: [batch_size, tgt_len, d_model], dec_self_attn: [batch_size, n_heads, tgt_len, tgt_len]
        dec_outputs, dec_self_attn = self.dec_self_attn(dec_inputs, dec_inputs, dec_inputs, dec_self_attn_mask)
        # dec_outputs: [batch_size, tgt_len, d_model], dec_enc_attn: [batch_size, h_heads, tgt_len, src_len]
        dec_outputs, dec_enc_attn = self.dec_enc_attn(dec_outputs, enc_outputs, enc_outputs, dec_enc_attn_mask)
        dec_outputs = self.pos_ffn(dec_outputs)  # [batch_size, tgt_len, d_model]
        return dec_outputs, dec_self_attn, dec_enc_attn


class Encoder(nn.Module):
    def __init__(self, d_model, n_heads, n_layers,d_k,d_v, d_feedforward, src_vocab_size):
        super(Encoder, self).__init__()
        self.src_emb = nn.Embedding(src_vocab_size, d_model)
        self.pos_emb = PositionalEncoding(d_model)
        self.layers = nn.ModuleList([EncoderLayer(d_model=d_model, n_heads=n_heads,d_k=d_k,d_v=d_v, d_feedforward=d_feedforward) for _ in range(n_layers)])

    def forward(self, enc_inputs):
        '''
        :param enc_inputs: [batch_size, src_len]
        '''
        enc_outputs = self.src_emb(enc_inputs)  # [batch_size, src_len, d_model]
        embeddings.append(self.src_emb.weight.cpu().detach().numpy())
        enc_outputs = self.pos_emb(enc_outputs.transpose(0, 1)).transpose(0, 1)  # [batch_size, src_len, d_model]
        enc_self_attn_mask = get_attn_pad_mask(enc_inputs, enc_inputs)  # [batch_size, src_len, src_len]
        enc_self_attns = []
        for layer in self.layers:
            # enc_outputs: [batch_size, src_len, d_model], enc_self_attn: [batch_size, n_heads, src_len, src_len]
            enc_outputs, enc_self_attn = layer(enc_outputs, enc_self_attn_mask)
            enc_self_attns.append(enc_self_attn)
        return enc_outputs, enc_self_attns


class Decoder(nn.Module):
    def __init__(self, d_model, n_heads, n_layers, d_k,d_v,d_feedforward, tgt_vocab_size):
        super(Decoder, self).__init__()
        self.tgt_emb = nn.Embedding(tgt_vocab_size, d_model)
        self.pos_emb = PositionalEncoding(d_model)
        self.layers = nn.ModuleList([DecoderLayer(d_model=d_model, n_heads=n_heads,d_k=d_k, d_v=d_v, d_feedforward=d_feedforward) for _ in range(n_layers)])

    def get_attn_subsequence_mask(self, seq):
        '''
        :param seq: [batch_size, tgt_len]
        '''
        attn_shape = [seq.size(0), seq.size(1), seq.size(1)]
        subsequence_mask = np.triu(np.ones(attn_shape), k=1)  # Upper triangular matrix
        subsequence_mask = torch.from_numpy(subsequence_mask).byte()
        return subsequence_mask  # [batch_size, tgt_len, tgt_len]

    def forward(self, dec_inputs, enc_inputs, enc_outputs):
        """

        :param dec_inputs: [batch_size, tgt_len]
        :param enc_inputs: [batch_size, src_len]
        :param enc_outputs: [batsh_size, src_len, d_model]
        :return:
        """
        dec_outputs = self.tgt_emb(dec_inputs).to(device)  # [batch_size, tgt_len, d_model]
        dec_outputs = self.pos_emb(dec_outputs.transpose(0, 1)).transpose(0, 1)  # [batch_size, tgt_len, d_model]
        dec_self_attn_pad_mask = get_attn_pad_mask(dec_inputs, dec_inputs)  # [batch_size, tgt_len, tgt_len]
        dec_self_attn_subsequence_mask = self.get_attn_subsequence_mask(dec_inputs).to(
            device)  # [batch_size, tgt_len, tgt_len]
        dec_self_attn_mask = torch.gt((dec_self_attn_pad_mask + dec_self_attn_subsequence_mask),
                                      0)  # [batch_size, tgt_len, tgt_len]

        dec_enc_attn_mask = get_attn_pad_mask(dec_inputs, enc_inputs)  # [batc_size, tgt_len, src_len]

        dec_self_attns, dec_enc_attns = [], []
        for layer in self.layers:
            # dec_outputs: [batch_size, tgt_len, d_model], dec_self_attn: [batch_size, n_heads, tgt_len, tgt_len], dec_enc_attn: [batch_size, h_heads, tgt_len, src_len]
            dec_outputs, dec_self_attn, dec_enc_attn = layer(dec_outputs, enc_outputs, dec_self_attn_mask,
                                                             dec_enc_attn_mask)
            dec_self_attns.append(dec_self_attn)
            dec_enc_attns.append(dec_enc_attn)
        return dec_outputs, dec_self_attns, dec_enc_attns


class MCFormer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=64, n_layers=3, n_heads=4,d_k=16,d_v=16, d_feedforward=512):
        super(MCFormer, self).__init__()
        self.encoder = Encoder(d_model=d_model, n_layers=n_layers, d_feedforward=d_feedforward, n_heads=n_heads,d_k=d_k,d_v=d_v, src_vocab_size=src_vocab_size).to(device)
        self.decoder = Decoder(d_model=d_model, n_layers=n_layers, d_feedforward=d_feedforward, n_heads=n_heads,d_k=d_k,d_v=d_v, tgt_vocab_size=tgt_vocab_size).to(device)
        self.projection = nn.Linear(d_model, tgt_vocab_size, bias=False)

    def forward(self, enc_inputs, dec_inputs):
        """
        :param enc_inputs: [batch_size, src_len]
        :param dec_inputs: [batch_size, tgt_len]
        """
        # tensor to store decoder outputs
        # outputs = torch.zeros(batch_size, tgt_len, tgt_vocab_size).to(self.device)

        # enc_outputs: [batch_size, src_len, d_model], enc_self_attns: [n_layers, batch_size, n_heads, src_len, src_len]
        enc_outputs, enc_self_attns = self.encoder(enc_inputs)
        # dec_outpus: [batch_size, tgt_len, d_model], dec_self_attns: [n_layers, batch_size, n_heads, tgt_len, tgt_len], dec_enc_attn: [n_layers, batch_size, tgt_len, src_len]
        dec_outputs, dec_self_attns, dec_enc_attns = self.decoder(dec_inputs, enc_inputs, enc_outputs)
        dec_logits = self.projection(dec_outputs)  # dec_logits: [batch_size, tgt_len, tgt_vocab_size]
        # dec_logits = nn.Softmax(dim=0)(dec_logits)
        return dec_logits.view(-1, dec_logits.size(-1)), enc_self_attns, dec_self_attns, dec_enc_attns


# # convert to the one-hot labels
# test_mask = to_categorical(test_mask, num_classes=4)
# converts an array into a one hot vector.
# Source: https://stackoverflow.com/questions/36960320/convert-a-2d-matrix-to-a-3d-one-hot-matrix-numpy
def one_hot(data: np.ndarray, ncols=2):
    shape = data.shape
    data = data.ravel()
    out = np.zeros((len(data), ncols))
    for i in range(len(data)):
        out[i, data[i]] = 1

    return out.reshape((shape[0], shape[1], ncols))

def test_currunt_model(settings):
    print("=====Start Test=====")
    mcformer_model = torch.load(os.path.join(settings['weights_dirpath']['mcformer'], settings['weights_name']['mcformer']))
    testdataset = ChannelDataset(settings['data_filepath']['test'],batch_length=1000,is_slide=False,transforms=None)
    testLoader = DataLoader(testdataset, batch_size=40, shuffle=False)

    batch_seq_length = 100
    score = 0
    total_num = 0
    mcformer_model.eval()
    for batch_x, batch_y in testLoader:
        batch_y = batch_y.reshape([-1, batch_seq_length])
        batch_x = batch_x.reshape([-1, batch_seq_length])
        batch_x = torch.cat((torch.ones(batch_x.size()[0]).reshape([-1, 1]), batch_x), dim=1)
        batch_y = torch.cat((torch.ones(batch_y.size()[0]).reshape([-1, 1]), batch_y), dim=1)
        Y = batch_y.detach().numpy()
        batch_y = batch_y.type(torch.IntTensor).to(device)
        batch_x = batch_x.type(torch.IntTensor).to(device)
        pre_y, enc_self_attns, dec_self_attns, enc_dec_attns = mcformer_model(batch_x, batch_y[:, :-1])
        # pre_y = torch.softmax(pre_y, dim=1)
        Y_pre = torch.argmax(pre_y, dim=1).cpu().detach().numpy()
        # score = 100 * np.mean(Y.reshape([-1]) == Y_pre)
        score += np.sum(Y[:, 1:].reshape([-1]) == Y_pre)
        total_num += len(Y_pre)
    score = score / total_num
    print("Test score: {:.6f}\nError rate: {:.6f}".format(score*100,1-score))
    return round(1-score,4)


def train_mcformer(settings,nlayers=3,nfeedforward=128):
    traindataset = ChannelDataset(settings['data_filepath']['train'], batch_length=1000, is_slide=True, transforms=None)
    trainLoader = DataLoader(traindataset, batch_size=60, shuffle=True)
    valdataset = ChannelDataset(settings['data_filepath']['val'], batch_length=1000, is_slide=True, transforms=None)
    valLoader = DataLoader(valdataset, batch_size=15, shuffle=False)

    # mcformer_model = MCFormer(src_vocab_size=150,tgt_vocab_size=2,d_model=512,n_layers=3,n_heads=8,d_k=64,d_v=64,d_feedforward=1024).to(device)
    mcformer_model = MCFormer(src_vocab_size=60, tgt_vocab_size=2, d_model=32, n_layers=3, n_heads=4, d_k=8,
                                    d_v=8, d_feedforward=128).to(device)
    # modelpath = os.path.join(settings['weights_dirpath'], settings['weights_name'])
    # mcformer_model = torch.load(modelpath)

    # criterion = nn.MSELoss()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(mcformer_model.parameters(), lr=0.001)
    # schedule = optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode="min",factor=0.1,patience=5,verbose=True)

    best_score = 0
    losses = []
    val_accs = []
    batch_seq_length = 100
    t = trange(50)
    for epoch in t:
        mcformer_model.train()
        loss_sum = 0
        for batch_x, batch_y in trainLoader:
            batch_y = batch_y.reshape([-1, batch_seq_length])
            # batch_x=torch.from_numpy(batch_x).cuda()
            batch_x = batch_x.reshape([-1, batch_seq_length])
            # batch_x[:, 0] = 1
            batch_x = torch.cat((torch.ones(batch_x.size()[0]).reshape([-1,1]),batch_x),dim=1)
            batch_y = torch.cat((torch.ones(batch_y.size()[0]).reshape([-1,1]),batch_y),dim=1)
            batch_x = batch_x.type(torch.LongTensor).to(device)
            batch_y = batch_y.type(torch.LongTensor).to(device)
            pre_y, _, _, _ = mcformer_model(batch_x, batch_y[:, :-1])
            # pre_y = torch.argmax(pre_y, dim=2)
            # tmp_y = one_hot(batch_y).reshape([-1,2])
            # tmp_y = torch.from_numpy(tmp_y).type(torch.IntTensor).to(device)
            # loss = criterion(pre_y, batch_y[:,:].reshape([-1]))
            loss = criterion(pre_y, batch_y[:, 1:].reshape([-1]))
            loss.backward()
            loss_sum += loss.item()
            optimizer.step()
            optimizer.zero_grad()
        losses.append(loss_sum)

        if epoch % 1 == 0:
            score = 0
            total_num = 0
            mcformer_model.eval()
            for batch_x, batch_y in valLoader:
                batch_y = batch_y.reshape([-1, batch_seq_length])
                batch_x = batch_x.reshape([-1, batch_seq_length])
                # batch_x[:, 0] = 1
                batch_x = torch.cat((torch.ones(batch_x.size()[0]).reshape([-1, 1]), batch_x), dim=1)
                batch_y = torch.cat((torch.ones(batch_y.size()[0]).reshape([-1, 1]), batch_y), dim=1)
                Y = batch_y.detach().numpy()
                batch_x = batch_x.type(torch.IntTensor).to(device)
                batch_y = batch_y.type(torch.IntTensor).to(device)
                pre_y, enc_self_attns, dec_self_attns, enc_dec_attns = mcformer_model(batch_x, batch_y[:, :-1])
                # pre_y = torch.softmax(pre_y, dim=1)
                Y_pre = torch.argmax(pre_y, dim=1).cpu().detach().numpy()
                # score = 100 * np.mean(Y.reshape([-1]) == Y_pre)
                # score += np.sum(Y.reshape([-1]) == Y_pre)
                score += np.sum(Y[:, 1:].reshape([-1]) == Y_pre)
                total_num += len(Y_pre)
            score = score / total_num
            val_accs.append(score * 100)
            if score > best_score:
                torch.save(mcformer_model, os.path.join(settings['weights_dirpath']['mcformer'], settings['weights_name']['mcformer']))
                best_score = score

                # schedule.step(1-score)

        t.set_postfix({'train_loss': "{:.6f}".format(loss.item()), 'val_accuracy': "{}".format(score * 100)})

    # np.save(os.path.join(settings['log_dirpath'], 'embeddings.npy'),np.array(embeddings))
    np.save(os.path.join(settings['log_dirpath']['train'], settings['weights_name']['mcformer']), np.array(losses))
    np.save(os.path.join(settings['log_dirpath']['val'], settings['weights_name']['mcformer']), np.array(val_accs))
    # seaborn.heatmap(enc_self_attns[0][0][0][:10, :10].cpu().detach().numpy())
    # plt.show()
    print("Best score: {:.6f}\nError rate: {:.6f}".format(best_score*100, 1 - best_score))

if __name__ == '__main__':
    # train_mcformer(yaml.safe_load(open("./settings/settings_aoji.yaml")),nlayers=3)
    test_currunt_model(yaml.safe_load(open("./settings/settings_aoji.yaml")))
