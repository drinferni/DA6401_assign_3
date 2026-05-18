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
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


# ══════════════════════════════════════════════════════════════════════
#   STANDALONE ATTENTION FUNCTION  
# ══════════════════════════════════════════════════════════════════════

def scaled_dot_product_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    d_k = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == True, -1e9)
    attn_w = F.softmax(scores, dim=-1)
    output = torch.matmul(attn_w, V)
    return output, attn_w


# ══════════════════════════════════════════════════════════════════════
# ❷  MASK HELPERS 
# ══════════════════════════════════════════════════════════════════════

def make_src_mask(src: torch.Tensor, pad_idx: int = 1) -> torch.Tensor:
    return (src == pad_idx).unsqueeze(1).unsqueeze(2)

def make_tgt_mask(tgt: torch.Tensor, pad_idx: int = 1) -> torch.Tensor:
    pad_mask = (tgt == pad_idx).unsqueeze(1).unsqueeze(2)
    tgt_len = tgt.size(1)
    causal_mask = torch.triu(torch.ones((1, 1, tgt_len, tgt_len), device=tgt.device), diagonal=1).bool()
    return pad_mask | causal_mask


# ══════════════════════════════════════════════════════════════════════
#  MULTI-HEAD ATTENTION 
# ══════════════════════════════════════════════════════════════════════

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, query, key, value, mask=None):
        nbatches = query.size(0)
        q = self.w_q(query).view(nbatches, -1, self.num_heads, self.d_k).transpose(1, 2)
        k = self.w_k(key).view(nbatches, -1, self.num_heads, self.d_k).transpose(1, 2)
        v = self.w_v(value).view(nbatches, -1, self.num_heads, self.d_k).transpose(1, 2)
        x, self.attn = scaled_dot_product_attention(q, k, v, mask=mask)
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.d_model)
        return self.w_o(x)


# ══════════════════════════════════════════════════════════════════════
#   POSITIONAL ENCODING  
# ══════════════════════════════════════════════════════════════════════

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


# ══════════════════════════════════════════════════════════════════════
#  FEED-FORWARD & LAYERS (Sub-components)
# ══════════════════════════════════════════════════════════════════════

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.linear2(self.dropout(F.relu(self.linear1(x))))

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ff = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        x = self.norm1(x + self.dropout(self.self_attn(x, x, x, mask)))
        return self.norm2(x + self.dropout(self.ff(x)))

class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ff = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, memory, src_mask, tgt_mask):
        x = self.norm1(x + self.dropout(self.self_attn(x, x, x, tgt_mask)))
        x = self.norm2(x + self.dropout(self.cross_attn(x, memory, memory, src_mask)))
        return self.norm3(x + self.dropout(self.ff(x)))

class Encoder(nn.Module):
    def __init__(self, layer, N):
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(N)])
        self.norm = nn.LayerNorm(layer.norm1.normalized_shape)

    def forward(self, x, mask):
        for layer in self.layers: x = layer(x, mask)
        return self.norm(x)

class Decoder(nn.Module):
    def __init__(self, layer, N):
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(N)])
        self.norm = nn.LayerNorm(layer.norm1.normalized_shape)

    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers: x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)


# ══════════════════════════════════════════════════════════════════════
#   FULL TRANSFORMER  
# ══════════════════════════════════════════════════════════════════════

class Transformer(nn.Module):
    def __init__(
        self,
        src_vocab_size: int = 8000,   # Example default
        tgt_vocab_size: int = 8000,   # Example default
        d_model:   int   = 512,
        N:         int   = 6,
        num_heads: int   = 8,
        d_ff:      int   = 2048,
        dropout:   float = 0.1,
        checkpoint_path: str = "model_weights.pth",
        vocab_path: str = "vocab.pt"
    ) -> None:
        super().__init__()
        
        # 1. Loading Vocabulary & Tokenizers inside Init
        # Replace these IDs with your actual Google Drive IDs
        if not os.path.exists(vocab_path):
            gdown.download(id="YOUR_VOCAB_FILE_ID", output=vocab_path, quiet=False)
        
        # Assume vocab file is a saved dict containing 'src_vocab' and 'tgt_inv_vocab'
        vocab_data = torch.load(vocab_path, map_location='cpu')
        self.src_vocab = vocab_data['src_vocab']
        self.tgt_inv_vocab = vocab_data['tgt_inv_vocab']
        
        # Adjust sizes to actual vocab length
        src_vocab_size = len(self.src_vocab)
        tgt_vocab_size = len(self.tgt_inv_vocab)

        try:
            self.spacy_de = spacy.load("de_core_news_sm")
        except:
            os.system("python -m spacy download de_core_news_sm")
            self.spacy_de = spacy.load("de_core_news_sm")

        # 2. Architecture Setup
        self.d_model = d_model
        self.src_embed = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embed = nn.Embedding(tgt_vocab_size, d_model)
        self.pos_enc = PositionalEncoding(d_model, dropout)
        self.encoder = Encoder(EncoderLayer(d_model, num_heads, d_ff, dropout), N)
        self.decoder = Decoder(DecoderLayer(d_model, num_heads, d_ff, dropout), N)
        self.fc_out = nn.Linear(d_model, tgt_vocab_size)

        # 3. Load Weights inside Init
        if not os.path.exists(checkpoint_path):
            gdown.download(id="YOUR_MODEL_WEIGHTS_ID", output=checkpoint_path, quiet=False)
        
        state_dict = torch.load(checkpoint_path, map_location='cpu')
        # Handle if saved within a 'model_state_dict' wrapper
        if 'model_state_dict' in state_dict:
            self.load_state_dict(state_dict['model_state_dict'])
        else:
            self.load_state_dict(state_dict)

    def encode(self, src, src_mask):
        return self.encoder(self.pos_enc(self.src_embed(src) * math.sqrt(self.d_model)), src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.fc_out(self.decoder(self.pos_enc(self.tgt_embed(tgt) * math.sqrt(self.d_model)), memory, src_mask, tgt_mask))

    def forward(self, src, tgt, src_mask, tgt_mask):
        return self.decode(self.encode(src, src_mask), src_mask, tgt, tgt_mask)

    def infer(self, german_sentence: str) -> str:
        """
        Executes end-to-end NLP processing:
        Tokenize -> Numericalize -> Transformer Inference -> Detokenize
        """
        self.eval()
        device = next(self.parameters()).device
        
        # 1. Tokenize & Numericalize (German)
        tokens = [tok.text.lower() for tok in self.spacy_de.tokenizer(german_sentence)]
        # Add <sos> and <eos> (assuming 2 and 3)
        src_indices = [2] + [self.src_vocab.get(t, 0) for t in tokens] + [3]
        src_tensor = torch.LongTensor([src_indices]).to(device)
        src_mask = make_src_mask(src_tensor)
        
        # 2. Forward Pass / Autoregressive Decoding
        with torch.no_grad():
            memory = self.encode(src_tensor, src_mask)
            ys = torch.ones(1, 1).fill_(2).type(torch.long).to(device) # <sos>
            
            for _ in range(100): # max_len
                out = self.decode(memory, src_mask, ys, make_tgt_mask(ys).to(device))
                prob = out[:, -1]
                _, next_word = torch.max(prob, dim=1)
                next_word = next_word.item()
                ys = torch.cat([ys, torch.ones(1, 1).type(torch.long).fill_(next_word).to(device)], dim=1)
                if next_word == 3: # <eos>
                    break
        
        # 3. Detokenize (English)
        decoded_indices = ys.squeeze().tolist()
        # Remove <sos>, <eos>, <pad>
        english_tokens = [self.tgt_inv_vocab[idx] for idx in decoded_indices if idx not in [1, 2, 3]]
        return " ".join(english_tokens)