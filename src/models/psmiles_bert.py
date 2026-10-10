"""Configurable BERT-style PSMILES encoder."""
from __future__ import annotations
from dataclasses import dataclass
import torch
from torch import nn

@dataclass
class PSmilesBertConfig:
    vocab_size:int
    max_sequence_length:int=192
    layers:int=8
    hidden_size:int=512
    attention_heads:int=8
    intermediate_size:int=2048
    dropout:float=0.1

class PSmilesBert(nn.Module):
    def __init__(self,cfg:PSmilesBertConfig):
        super().__init__(); self.cfg=cfg
        self.token=nn.Embedding(cfg.vocab_size,cfg.hidden_size,padding_idx=0)
        self.pos=nn.Embedding(cfg.max_sequence_length,cfg.hidden_size)
        layer=nn.TransformerEncoderLayer(d_model=cfg.hidden_size,nhead=cfg.attention_heads,dim_feedforward=cfg.intermediate_size,dropout=cfg.dropout,activation='gelu',batch_first=True,norm_first=True)
        self.encoder=nn.TransformerEncoder(layer,num_layers=cfg.layers)
        self.norm=nn.LayerNorm(cfg.hidden_size)
    def forward(self,input_ids,attention_mask=None):
        b,l=input_ids.shape; pos=torch.arange(l,device=input_ids.device).unsqueeze(0).expand(b,l)
        x=self.token(input_ids)+self.pos(pos)
        key_padding_mask=(attention_mask==0) if attention_mask is not None else None
        x=self.encoder(x,src_key_padding_mask=key_padding_mask)
        x=self.norm(x)
        if attention_mask is None: return x, x[:,0]
        mask=attention_mask.unsqueeze(-1).float(); pooled=(x*mask).sum(1)/mask.sum(1).clamp_min(1)
        return x, pooled
class MLMHead(nn.Module):
    def __init__(self,hidden,vocab): super().__init__(); self.net=nn.Sequential(nn.Linear(hidden,hidden),nn.GELU(),nn.LayerNorm(hidden),nn.Linear(hidden,vocab))
    def forward(self,x): return self.net(x)
class RegressionHead(nn.Module):
    def __init__(self,hidden,out=1): super().__init__(); self.net=nn.Sequential(nn.LayerNorm(hidden),nn.Linear(hidden,hidden),nn.GELU(),nn.Dropout(0.1),nn.Linear(hidden,out))
    def forward(self,x): return self.net(x)
