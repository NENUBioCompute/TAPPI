#!source /usr/local/Ascend/ascend-toolkit/set_env.sh
import torch

import torch.nn as nn
import math
from functools import partial
from esm.modules import ContactPredictionHead, ESM1bLayerNorm, RobertaLMHead, TransformerLayer, MultiheadAttention 
import esm
from typing import Union

class GroundingAttention(nn.Module):
    def __init__(self, dim, num_heads=4, qkv_bias=True,
                 attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.kv = nn.Linear(dim, dim*2, bias=qkv_bias)
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        # self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, r):
        B, N, C = x.shape
        B_, N_, C_ = r.shape

        kv = self.kv(r).reshape(B_, N_, 2, self.num_heads, C_ //
                                self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv.unbind(0)
        q = self.q(x).reshape(B, N, self.num_heads, C //
                              self.num_heads).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, heads, N, N_)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        # x = self.proj(x)
        x = self.proj_drop(x)
        return x

class FFN(nn.Module):
    """
    Residual feed-forward network.
    """
    
    def __init__(self, input_dim, hidden_dim):
        super(FFN, self).__init__()
        self.relu = nn.ReLU()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):

        residual = x

        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)

        x = residual + x
        return x

class bertlayer(nn.Module):
    def __init__(self, embeddingdim, hidden_dim, num_head = 16):
        super(bertlayer, self).__init__()
        self.atte_norm = ESM1bLayerNorm(embeddingdim)
        self.ffn_norm = ESM1bLayerNorm(embeddingdim)
        self.atte = torch.nn.MultiheadAttention(embed_dim = embeddingdim, num_heads = num_head, dropout = 0.0)
        self.ffn = FFN(embeddingdim, hidden_dim)
    def forward(self, x, x_padding_mask):
        residual = x
        x = self.atte_norm(x)
        x, _ = self.atte(x, x, x, key_padding_mask = x_padding_mask)
        x = x + residual
        x = x + self.ffn(self.ffn_norm(x))
        return x


class InteractionBlock(nn.Module):
    def __init__(self, embed_dim, ffn_dim, BertLayerNorm = ESM1bLayerNorm, attention_heads = 16, add_bias_kv = True, use_rotary_embeddings = True):
        super(InteractionBlock, self).__init__()
        # self.injector_query_norm = norm_layer(embedding_dim)
        # self.injector_kv_norm = norm_layer(embedding_dim)
        # self.extractor_query_norm = norm_layer(embedding_dim)
        # self.extractor_kv_norm = norm_layer(embedding_dim)
        # self.extractor_norm = norm_layer(embedding_dim)
        # self.injector = GroundingAttention(embedding_dim)
        # self.block = GroundingAttention(embedding_dim)
        # self.extractor = GroundingAttention(embedding_dim)
        # self.extractor_ffn = FFN(embedding_dim * ffn_dim, ffn_dim_rate * embedding_dim * ffn_dim)
        self.attention_heads = attention_heads
        self.embed_dim = embed_dim
        self.ffn_dim = ffn_dim
        self.injector_q_norm = BertLayerNorm(embed_dim)
        self.injector_kv_norm = BertLayerNorm(embed_dim)
        # self.injector = GroundingAttention(embed_dim)

        self.injector = torch.nn.MultiheadAttention(embed_dim = embed_dim, num_heads = attention_heads, dropout = 0.0)
        self.block = bertlayer(embed_dim, embed_dim * 4)
        self.extractor_q_norm = BertLayerNorm(embed_dim)
        self.extractor_kv_norm = BertLayerNorm(embed_dim)
        self.extractor = torch.nn.MultiheadAttention(embed_dim = embed_dim, num_heads = attention_heads, dropout = 0.0)
        self.ffn = FFN(embed_dim, ffn_dim)
    def forward(self, x, r, x_attn_padding_mask=None, r_attn_padding_mask = None, need_head_weights=False):
        # x = self.injector(self.injector_query_norm(x), self.injector_kv_norm(r)) + x
        # x = self.block(x, x)
        # r = self.extractor(self.extractor_query_norm(r), self.extractor_kv_norm(x)) + r
        # r = r + self.extractor_ffn(self.extractor_norm(r))
        
        
        # x, _ = self.injector_attention(
        #     query=self.injector_q_norm(x),
        #     key=self.injector_kv_norm(r),
        #     value=self.injector_kv_norm(r),
        #     key_padding_mask=self_attn_padding_mask,
        #     need_weights=True,
        #     need_head_weights=need_head_weights,
        #     attn_mask=self_attn_mask,
        # )
        # print(self.injector_q_norm(x).shape)
        # print(self.injector_q_norm(r).shape)
        
        
        # x = x + self.injector(self.injector_q_norm(x),self.injector_kv_norm(r))
        residual_x = x
        residual_r = r
        # x = self.injector_q_norm(x)
        # r = self.injector_kv_norm(r)
        x = x.transpose(0, 1)
        r = r.transpose(0, 1)
        x = self.injector_q_norm(x)
        r = self.injector_kv_norm(r)
        # print("r.shape")
        # print(r.shape)
        # print("r_attn_padding_mask.shape")
        # print(r_attn_padding_mask.shape)
        # print("x.shape")
        # print(x.shape)
        x, attn = self.injector(x, r, r, key_padding_mask=r_attn_padding_mask )
        x = x.transpose(0, 1)
        x = x + residual_x
        r = residual_r
        x = x.transpose(0, 1)
        x = self.block(x, x_attn_padding_mask)
        
        # r = r + self.extractor(self.extractor_q_norm(r),self.extractor_kv_norm(x))
        residual_r = r
        residual_x = x.transpose(0, 1)
        # x = self.extractor_kv_norm(x)
        # r = self.extractor_q_norm(r)
        # x = x.transpose(0, 1)
        r = r.transpose(0, 1)
        # print(r.shape)
        # print(x.shape)
        # print(x_attn_padding_mask.shape)
        x = self.extractor_kv_norm(x)
        r = self.extractor_q_norm(r)
        r, attn = self.extractor(r, x, x, key_padding_mask=x_attn_padding_mask)
        r = r.transpose(0, 1)
        r = r + residual_r
        
        r = self.ffn(r)
        return residual_x, r

import esm
class DynamicFeatureSelector(nn.Module):
    def __init__(self, input_size, num_features, num_layers=4):
        super(DynamicFeatureSelector, self).__init__()
        

        self.hidden_layers = []
        self.batch_norm_layers = []  
        current_size = input_size
        decrement = (input_size - num_features) // num_layers  

        for _ in range(num_layers):
            if current_size <= num_features:
                break
            next_size = max(current_size - decrement, num_features)
            self.hidden_layers.append(nn.Linear(current_size, next_size))
            self.batch_norm_layers.append(nn.BatchNorm1d(next_size))  # 添加 BatchNorm 层
            current_size = next_size


        self.hidden_layers = nn.ModuleList(self.hidden_layers)
        self.batch_norm_layers = nn.ModuleList(self.batch_norm_layers)
        self.output_layer = nn.Linear(current_size, num_features)  # 输出层

    def forward(self, x):
        for layer, batch_norm in zip(self.hidden_layers, self.batch_norm_layers):
            x = layer(x)
            x = batch_norm(x)
            x = torch.relu(x)
        x = self.output_layer(x)
        return x

class mpi_adapter(nn.Module):
    def __init__(self, embedding_dim, ffn_dim, num_layers =39):
        super(mpi_adapter, self).__init__()
        # self.embedding = nn.Embedding(num_embeddings=22, embedding_dim=embedding_dim)
        # self.par_position = PositionalEncoding(num_hiddens=embedding_dim, max_len = par_len)
        # self.mut_position = PositionalEncoding(num_hiddens=embedding_dim, max_len = mut_len)
        # self.mut_esm, _ = esm.pretrained.esm2_t33_650M_UR50D()
        # _, self.alphabet = esm.pretrained.esm2_t33_650M_UR50D()
        # self.padding_idx = self.alphabet.padding_idx
        # self.esm.to(self.device)
        # self.esm.eval()
        self.liner_mut = nn.Linear(1280,embedding_dim)
        self.liner_par = nn.Linear(1280,embedding_dim)
        # self.batch_converter = self.alphabet.get_batch_converter()
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(InteractionBlock(embedding_dim, ffn_dim))
    def forward(self, mut0, mut1, par, mut0_padding_mask, par_padding_mask):
        # print([(str(0), sequence) for sequence in mut0s])
        # print([(str(0), sequence) for sequence in mut1s])
        # print([(str(0), sequence) for sequence in pars])
        # _, _, mut0 = self.batch_converter([(str(0), sequence) for sequence in mut0s])
        # _, _, mut1 = self.batch_converter([(str(0), sequence) for sequence in mut1s])
        # _, _, par = self.batch_converter([(str(0), sequence) for sequence in pars])
        # print(par)
        # mut0 = mut0s.to(self.device)
        # mut1 = mut1s.to(self.device)
        # par = pars.to(self.device)
        # mut0_padding_mask = mut0_padding_mask.to(self.device)
        # par_padding_mask = par_padding_mask.to(self.device)
        # mut0_padding_mask = mut0.eq(self.padding_idx)
        # mut1_padding_mask = mut1.eq(self.padding_idx)
        # par_padding_mask = par.eq(self.padding_idx)
        # mut0_padding_mask = torch.cat((mut0_padding_mask,mut1_padding_mask),dim=1)
        mut0 = torch.cat((mut0,mut1),dim=1)
        mut0 = self.liner_mut(mut0)
        par = self.liner_par(par)
        for layer in self.layers:
            mut0, par = layer(mut0, par, x_attn_padding_mask = mut0_padding_mask, r_attn_padding_mask = par_padding_mask)
        return par

class MLP_head(nn.Module):
    def __init__(self, embedding_dim, num_layers = 2):
        super(MLP_head, self).__init__()
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(GroundingAttention(embedding_dim))
    def forward(self, x):
        for layer in self.layers:
            x = layer(x, x)
        return x[:, 0, :]

class MLP_head_one_layer(nn.Module):
    def __init__(self, embedding_dim, num_layers = 1,attention_heads = 16):
        super(MLP_head_one_layer, self).__init__()
        self.layers = nn.ModuleList()
        self.attention = torch.nn.MultiheadAttention(embed_dim = embedding_dim, num_heads = attention_heads, dropout = 0.0)
        # for _ in range(num_layers):
        #     self.layers.append(TransformerLayer(
        #         embedding_dim,
        #         4 * embedding_dim, # embed_dim = 1280
        #         attention_heads, # 20
        #         add_bias_kv=False,
        #         use_esm1b_layer_norm=True,
        #         use_rotary_embeddings=True,
        #         ))
    def forward(self, x, x_attn_padding_mask, need_head_weights=False):
        result = x[:, 0:1, :]
        x = x.transpose(0, 1)
        result = result.transpose(0, 1)
        result, attn = self.attention(result, x, x, key_padding_mask = x_attn_padding_mask)
        # for layer in self.layers:
        #     x = x.transpose(0, 1)
        #     x, _ = layer(x,
        #                 self_attn_padding_mask=x_attn_padding_mask,
        #                 need_head_weights=need_head_weights,
        #                 )
        #     x = x.transpose(0, 1)
        result = result.transpose(0, 1)
        result = result.squeeze(1)
        return result

class TAPPI(nn.Module):
    def __init__(self, embedding_dim = 192, num_layers = 39):
        super(TAPPI, self).__init__()
        self.result = nn.Parameter(torch.randn(1280))
        self.backbone = mpi_adapter(embedding_dim, embedding_dim*4, num_layers = num_layers)
        self.neck = MLP_head_one_layer(embedding_dim)
        # self.head = cross_entropy_bert(embedding_dim, device)
        self.head = nn.Linear(embedding_dim, 4)
        
    def predict(self,  mut0s, mut1s, pars, mut0_padding_mask = None, par_padding_mask = None):
        res = self.result.repeat(pars.shape[0], 1).unsqueeze(1) 
        pars = torch.concat([res,pars], dim = 1)
        false_column = torch.zeros(par_padding_mask.size(0), 1, dtype=torch.bool, device=pars.device)

        par_padding_mask = torch.cat((false_column, par_padding_mask), dim=1)
        x = self.backbone(mut0s, mut1s, pars, mut0_padding_mask, par_padding_mask)

        # x = self.neck(pars, par_padding_mask)
        x = self.neck(x, par_padding_mask)

        # x = self.head(label, x, weight)
        x = self.head(x)
        return x

    
    def forward(self, mut0s, mut1s, pars, mut0_padding_mask, par_padding_mask, weight, label):

        mut0s = mut0s.to(pars.device)
        mut1s = mut1s.to(pars.device)
        pars = pars.to(pars.device)
        mut0_padding_mask = mut0_padding_mask.to(pars.device)
        par_padding_mask = par_padding_mask.to(pars.device)

        res = self.result.repeat(pars.shape[0], 1).unsqueeze(1)
        pars = torch.cat([res, pars], dim=1)

        false_column = torch.zeros(par_padding_mask.size(0), 1, dtype=torch.bool).to(pars.device)
        par_padding_mask = torch.cat((false_column, par_padding_mask), dim=1)

        x = self.backbone(mut0s, mut1s, pars, mut0_padding_mask, par_padding_mask)
        x_neck = self.neck(x, par_padding_mask)
        x = self.head(x_neck)
        return x, x_neck, None, None
