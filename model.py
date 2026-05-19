"""
model.py — Transformer Architecture Implementation
DA6401 Assignment 3: "Attention Is All You Need"

AUTOGRADER CONTRACT (DO NOT MODIFY SIGNATURES):
  ┌─────────────────────────────────────────────────────────────────┐
  │  scaled_dot_product_attention(Q, K, V, mask) → (out, weights)  │
  │  MultiHeadAttention.forward(q, k, v, mask)   → Tensor          │
  │  PositionalEncoding.forward(x)               → Tensor          │
  │  make_src_mask(src, pad_idx)                 → BoolTensor      │
  │  make_tgt_mask(tgt, pad_idx)                 → BoolTensor      │
  │  Transformer.encode(src, src_mask)           → Tensor          │
  │  Transformer.decode(memory,src_m,tgt,tgt_m)  → Tensor          │
  └─────────────────────────────────────────────────────────────────┘
"""

import math
import copy
import os
import gdown
import spacy
import spacy.cli
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

# ══════════════════════════════════════════════════════════════════════
#   STANDALONE ATTENTION & MASK HELPERS 
# ══════════════════════════════════════════════════════════════════════



def scaled_dot_product_attention(Q, K, V, mask=None):
    d_k = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == True, -1e9)
    attn_w = F.softmax(scores, dim=-1)
    return torch.matmul(attn_w, V), attn_w

def make_src_mask(src, pad_idx=1):
    return (src == pad_idx).unsqueeze(1).unsqueeze(2)

def make_tgt_mask(tgt, pad_idx=1):
    pad_mask = (tgt == pad_idx).unsqueeze(1).unsqueeze(2)
    sz = tgt.size(1)
    causal_mask = torch.triu(torch.ones((1, 1, sz, sz), device=tgt.device), 1).bool()
    return pad_mask | causal_mask

# ══════════════════════════════════════════════════════════════════════
#  TRANSFORMER COMPONENTS
# ══════════════════════════════════════════════════════════════════════

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        self.d_model, self.num_heads, self.d_k = d_model, num_heads, d_model // num_heads
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
    def forward(self, q, k, v, mask=None):
        bs = q.size(0)
        q = self.w_q(q).view(bs, -1, self.num_heads, self.d_k).transpose(1, 2)
        k = self.w_k(k).view(bs, -1, self.num_heads, self.d_k).transpose(1, 2)
        v = self.w_v(v).view(bs, -1, self.num_heads, self.d_k).transpose(1, 2)
        x, _ = scaled_dot_product_attention(q, k, v, mask)
        return self.w_o(x.transpose(1, 2).contiguous().view(bs, -1, self.d_model))

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2], pe[:, 1::2] = torch.sin(pos * div), torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))
        self.dropout = nn.Dropout(dropout)
    def forward(self, x): return self.dropout(x + self.pe[:, :x.size(1)])

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.l1, self.l2, self.dropout = nn.Linear(d_model, d_ff), nn.Linear(d_ff, d_model), nn.Dropout(dropout)
    def forward(self, x): return self.l2(self.dropout(F.relu(self.l1(x))))

class EncoderLayer(nn.Module):
    def __init__(self, d_model, h, d_ff, drp):
        super().__init__()
        self.attn, self.ff, self.n1, self.n2, self.drp = MultiHeadAttention(d_model, h, drp), PositionwiseFeedForward(d_model, d_ff, drp), nn.LayerNorm(d_model), nn.LayerNorm(d_model), nn.Dropout(drp)
    def forward(self, x, m):
        x = self.n1(x + self.drp(self.attn(x, x, x, m)))
        return self.n2(x + self.drp(self.ff(x)))

class DecoderLayer(nn.Module):
    def __init__(self, d_model, h, d_ff, drp):
        super().__init__()
        self.s_attn, self.c_attn, self.ff, self.n1, self.n2, self.n3, self.drp = MultiHeadAttention(d_model, h, drp), MultiHeadAttention(d_model, h, drp), PositionwiseFeedForward(d_model, d_ff, drp), nn.LayerNorm(d_model), nn.LayerNorm(d_model), nn.LayerNorm(d_model), nn.Dropout(drp)
    def forward(self, x, mem, sm, tm):
        x = self.n1(x + self.drp(self.s_attn(x, x, x, tm)))
        x = self.n2(x + self.drp(self.c_attn(x, mem, mem, sm)))
        return self.n3(x + self.drp(self.ff(x)))

class Encoder(nn.Module):
    def __init__(self, l, N):
        super().__init__()
        self.layers, self.norm = nn.ModuleList([copy.deepcopy(l) for _ in range(N)]), nn.LayerNorm(l.n1.normalized_shape)
    def forward(self, x, m):
        for l in self.layers: x = l(x, m)
        return self.norm(x)

class Decoder(nn.Module):
    def __init__(self, l, N):
        super().__init__()
        self.layers, self.norm = nn.ModuleList([copy.deepcopy(l) for _ in range(N)]), nn.LayerNorm(l.n1.normalized_shape)
    def forward(self, x, mem, sm, tm):
        for l in self.layers: x = l(x, mem, sm, tm)
        return self.norm(x)

# ══════════════════════════════════════════════════════════════════════
#   FULL TRANSFORMER  
# ══════════════════════════════════════════════════════════════════════

class Transformer(nn.Module):
    def __init__(self,src_vocab_size: Optional[int] = None,tgt_vocab_size: Optional[int] = None,d_model= 512,N = 6,num_heads = 8,d_ff = 2048,dropout = 0.1,checkpoint_path: str = "checkpoint.pt") :
        super().__init__()
        
        self.src_vocab = {}
        self.tgt_inv_vocab = {}
        self.d_model = d_model
        
        # Default sizes if no checkpoint and no args
        s_size = src_vocab_size if src_vocab_size else 7853
        t_size = tgt_vocab_size if tgt_vocab_size else 5893

        try:
            if not os.path.exists(checkpoint_path):
                gdown.download(id="1C8YWjczTenNdRDbfRYWvioNndc1zM_24", output=checkpoint_path, quiet=False)
            
            ckpt = torch.load(checkpoint_path, map_location='cpu')
            self.src_vocab = ckpt.get('src_vocab', {})
            self.tgt_inv_vocab = ckpt.get('tgt_inv_vocab', {})
            
            if self.src_vocab: s_size = len(self.src_vocab)
            if self.tgt_inv_vocab: t_size = len(self.tgt_inv_vocab)
            print("Checkpoint found. Loading architecture with extracted vocab sizes.")
        except Exception:
            print("No checkpoint found or download failed. Initializing with provided/default sizes.")

        self.src_embed = nn.Embedding(s_size, d_model)
        self.tgt_embed = nn.Embedding(t_size, d_model)
        self.pos_enc = PositionalEncoding(d_model, dropout)
        self.encoder = Encoder(EncoderLayer(d_model, num_heads, d_ff, dropout), N)
        self.decoder = Decoder(DecoderLayer(d_model, num_heads, d_ff, dropout), N)
        self.fc_out = nn.Linear(d_model, t_size)

        if 'ckpt' in locals() and 'model_state_dict' in ckpt:
            self.load_state_dict(ckpt['model_state_dict'])
            print("Weights loaded successfully.")

        import spacy.cli
        try:
            self.spacy_de = spacy.load("de_core_news_sm")
        except:
            try:
                spacy.cli.download("de_core_news_sm")
                self.spacy_de = spacy.load("de_core_news_sm")
            except:
                print("Warning: Could not download de_core_news_sm. Falling back to blank 'de' model.")
                self.spacy_de = spacy.blank("de")

    def encode(self, src, src_mask):
        return self.encoder(self.pos_enc(self.src_embed(src) * math.sqrt(self.d_model)), src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.fc_out(self.decoder(self.pos_enc(self.tgt_embed(tgt) * math.sqrt(self.d_model)), memory, src_mask, tgt_mask))

    def forward(self, src, tgt, src_mask, tgt_mask):
        return self.decode(self.encode(src, src_mask), src_mask, tgt, tgt_mask)
    
    def greedy_decode(self,model,src,src_mask,max_len,start_symbol,end_symbol,device= "cpu") :
        model.eval()
        memory = model.encode(src, src_mask)
        ys = torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(device)
        
        for _ in range(max_len - 1):
            tgt_mask = make_tgt_mask(ys).to(device)
            out = model.decode(memory, src_mask, ys, tgt_mask)
            prob = out[:, -1]
            _, next_word = torch.max(prob, dim=1)
            next_word = next_word.item()
            
            ys = torch.cat([ys, torch.ones(1, 1).type(torch.long).fill_(next_word).to(device)], dim=1)
            if next_word == end_symbol:
                break
        return ys


    def infer(self, german_sentence) :
        self.eval()
        device = next(self.parameters()).device
        
        tokens = [tok.text.lower() for tok in self.spacy_de.tokenizer(german_sentence)]
        
        src_indices = [2] + [self.src_vocab.get(t, 0) for t in tokens] + [3]
        src_tensor = torch.LongTensor([src_indices]).to(device)

        src_mask = make_src_mask(src_tensor).to(device)
    
        with torch.no_grad():
            res_indices_tensor = self.greedy_decode(model=self, src=src_tensor, src_mask=src_mask, max_len=1000, start_symbol=2, end_symbol=3, device=device)

        indices = res_indices_tensor.squeeze().tolist()
        if not isinstance(indices, list):
            indices = [indices]
        return " ".join([self.tgt_inv_vocab.get(idx, str(idx)) for idx in indices if idx not in [1, 2, 3]])