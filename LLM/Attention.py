import torch
import torch.nn as nn
import math

class MHA(nn.Module):
    """
    Multi head attention module
    """
    def __init__(self, hidden_dim: int,
                 num_heads: int,
                 dropout: float =0.1,
                 need_weights: bool = False):
        super().__init__()
        assert hidden_dim % num_heads == 0

        self.qkv_project = nn.Linear(hidden_dim, hidden_dim * 3)
        self.out_project = nn.Linear(hidden_dim, hidden_dim)

        self.head_dim = hidden_dim // num_heads
        self.num_heads = num_heads
        self.dropout = nn.Dropout(dropout)
        self.need_weights = need_weights

    def forward(self, x, mask=None): # (B, S, D), (B, S) or (B, S, S)
        B, S, _ = x.shape
        qkv = self.qkv_project(x)
        q, k, v = torch.chunk(qkv, 3, dim=-1) # (B, S, D)

        q = q.reshape(B, S, self.num_heads, self.head_dim).transpose(1, 2).contiguous() # (B, H, S, D/H)
        k = k.reshape(B, S, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
        v = v.reshape(B, S, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

        weights = (q @ k.permute(0, 1, 3, 2)) / math.sqrt(self.head_dim)
        if mask is not None:
            mask = torch.unsqueeze(mask, dim=1)
            weights = torch.masked_fill(weights, mask, float('-inf'))
        weights = torch.softmax(weights, dim=-1) # (B, H, S, S)
        weights = self.dropout(weights)

        v = torch.transpose(weights @ v, 1, 2).contiguous() # (B, S, H, D/H)
        v = v.view(B, S, -1) # (B, S, D)
        v = self.out_project(v)

        if self.need_weights:
            return v, weights
        return v

class GQA(nn.Module):
    """
    Group query attention
    """
    def __init__(self, hidden_dim, num_heads, num_groups, dropout, need_weights):
        super().__init__()
        assert num_heads % num_groups == 0
        assert hidden_dim % num_heads == 0

        self.head_dim = hidden_dim // num_heads
        self.num_heads_q = num_heads
        self.num_heads_kv = num_heads // num_groups
        self.num_groups = num_groups

        self.q_project = nn.Linear(hidden_dim, self.head_dim * self.num_heads_q)
        self.k_project = nn.Linear(hidden_dim, self.head_dim * self.num_heads_kv)
        self.v_project = nn.Linear(hidden_dim, self.head_dim * self.num_heads_kv)
        self.o_project = nn.Linear(hidden_dim, hidden_dim)

        self.dropout = nn.Dropout(dropout)
        self.need_weights = need_weights
    
    def forward(self, x, mask=None):
        B, S, _ = x.shape
        q = self.q_project(x)
        k = self.k_project(x)
        v = self.v_project(x)
        
        q = q.reshape(B, S, self.num_heads_q, self.head_dim).transpose(1, 2)
        k = k.reshape(B, S, self.num_heads_kv, self.head_dim).transpose(1, 2)
        v = v.reshape(B, S, self.num_heads_kv, self.head_dim).transpose(1, 2)
        
        weights = (q @ k.transpose(2, 3).repeat_interleave(self.num_groups, dim=1)) / math.sqrt(self.head_dim)
        if mask is not None:
            mask = torch.unsqueeze(mask, dim=1)
            weights = torch.masked_fill(weights, mask, float('-inf'))
        weights = torch.softmax(weights, dim=-1)
        weights = self.dropout(weights)
        
        v = (weights @ v.repeat_interleave(self.num_groups, dim=1)).transpose(1, 2).contiguous()
        v = v.view(B, S, -1)
        v= self.o_project(v)
        
        if self.need_weights:
            return v, weights
        return v

if __name__ == '__main__':
    mha = GQA(12, 4, 2, 0.1, True)
    x = torch.rand(4, 6, 12)
    o, w = mha(x)