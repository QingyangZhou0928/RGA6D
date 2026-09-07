# ------------------------------------------------------------------------------------
# Modified from GeoTransformer (https://github.com/qinzheng93/GeoTransformer)
# Originally authored by Zheng Qin (Copyright (c) 2022 Zheng Qin)
# Licensed under the MIT License.
# Modifications copyright (c) 2026 Qingyang Zhou
# ------------------------------------------------------------------------------------
r"""Transformer with Relative Positional Embeddings.

Relative positional embedding is further projected in each multi-head attention layer.

The shape of input tensor should be (B, N, C). Implemented with `nn.Linear` and `nn.LayerNorm` (with affine).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from IPython import embed

from models.modules.layers import build_dropout_layer
from models.modules.transformer.output_layer import AttentionOutput


class RPEMultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=None):
        super(RPEMultiHeadAttention, self).__init__()
        if d_model % num_heads != 0:
            raise ValueError('`d_model` ({}) must be a multiple of `num_heads` ({}).'.format(d_model, num_heads))

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_model_per_head = d_model // num_heads

        self.proj_q = nn.Linear(self.d_model, self.d_model)
        self.proj_k = nn.Linear(self.d_model, self.d_model)
        self.proj_v = nn.Linear(self.d_model, self.d_model)
        self.proj_p = nn.Linear(self.d_model, self.d_model)

        self.dropout = build_dropout_layer(dropout)

    def forward(self, input_q, input_k, input_v, embed_qk, key_weights=None, key_masks=None, attention_factors=None):
        r"""Scaled Dot-Product Attention with Pre-computed Relative Positional Embedding (forward)

        Args:
            input_q: torch.Tensor (B, N, C)
            input_k: torch.Tensor (B, M, C)
            input_v: torch.Tensor (B, M, C)
            embed_qk: torch.Tensor (B, N, M, C), relative positional embedding
            key_weights: torch.Tensor (B, M), soft masks for the keys
            key_masks: torch.Tensor (B, M), True if ignored, False if preserved
            attention_factors: torch.Tensor (B, N, M)

        Returns:
            hidden_states: torch.Tensor (B, C, N)
            attention_scores: torch.Tensor (B, H, N, M)
        """
        q = rearrange(self.proj_q(input_q), 'b n (h c) -> b h n c', h=self.num_heads)
        k = rearrange(self.proj_k(input_k), 'b m (h c) -> b h m c', h=self.num_heads)
        v = rearrange(self.proj_v(input_v), 'b m (h c) -> b h m c', h=self.num_heads)
        p = rearrange(self.proj_p(embed_qk), 'b n m (h c) -> b h n m c', h=self.num_heads)

        attention_scores_p = torch.einsum('bhnc,bhnmc->bhnm', q, p)  # dot_product(q,p)
        attention_scores_e = torch.einsum('bhnc,bhmc->bhnm', q, k)  # dot_product(q,k)
        attention_scores = (attention_scores_e + attention_scores_p) / self.d_model_per_head ** 0.5
        if attention_factors is not None:
            attention_scores = attention_factors.unsqueeze(1) * attention_scores
        if key_weights is not None:
            attention_scores = attention_scores * key_weights.unsqueeze(1).unsqueeze(1)
        if key_masks is not None:  # changed (B, M) to (B, N, M)
            mask_extend = torch.ones(key_masks.size(0),1+key_masks.size(1)+key_masks.size(2),1+key_masks.size(1)+key_masks.size(2)).bool().to(key_masks.device)
            mask_extend[:,1:1+key_masks.size(1),1+key_masks.size(1):] = key_masks
            mask_extend[:,1+key_masks.size(1):,1:1+key_masks.size(1)] = key_masks.transpose(1,2)
            attention_scores = attention_scores.masked_fill(~mask_extend.unsqueeze(1), float('-inf'))
            # attention_scores = attention_scores.masked_fill(key_masks.unsqueeze(1).unsqueeze(1), float('-inf'))
        attention_scores = F.softmax(attention_scores, dim=-1)
        attention_scores = self.dropout(attention_scores)

        hidden_states = torch.matmul(attention_scores, v)  # (B,H,N,M)*(B,H,M,C)->(B,H,N,C)

        hidden_states = rearrange(hidden_states, 'b h n c -> b n (h c)')

        return hidden_states, attention_scores  # (B,H,N,C) (B,H,N,M)


class RPEAttentionLayer(nn.Module):
    def __init__(self, d_model, num_heads, dropout=None):
        super(RPEAttentionLayer, self).__init__()
        self.attention = RPEMultiHeadAttention(d_model, num_heads, dropout=dropout)
        self.linear = nn.Linear(d_model, d_model)
        self.dropout = build_dropout_layer(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        input_states,  # feats0
        memory_states,  # feats0
        position_states,  # embeddings0
        memory_weights=None,
        memory_masks=None,
        attention_factors=None,
    ):
        hidden_states, attention_scores = self.attention(
            input_states,  # feats0
            memory_states,  # feats0
            memory_states,  # feats0
            position_states,  # embeddings0
            key_weights=memory_weights,
            key_masks=memory_masks,
            attention_factors=attention_factors,
        )  # (B,H,N,C) (B,H,N,M)
        hidden_states = self.linear(hidden_states)
        hidden_states = self.dropout(hidden_states)
        output_states = self.norm(hidden_states + input_states)
        return output_states, attention_scores


class RPETransformerLayer(nn.Module):
    def __init__(self, d_model, num_heads, dropout=None, activation_fn='ReLU'):
        super(RPETransformerLayer, self).__init__()
        self.attention = RPEAttentionLayer(d_model, num_heads, dropout=dropout)
        self.output = AttentionOutput(d_model, dropout=dropout, activation_fn=activation_fn)

    def forward(
        self,
        input_states,  # feats0
        memory_states,  # feats0
        position_states,  # embeddings0
        memory_weights=None,
        memory_masks=None,
        attention_factors=None,
    ): 
        hidden_states, attention_scores = self.attention(
            input_states,  # feats0
            memory_states,  # feats0
            position_states,  # embeddings0
            memory_weights=memory_weights,
            memory_masks=memory_masks,
            attention_factors=attention_factors,
        )
        output_states = self.output(hidden_states)
        return output_states, attention_scores