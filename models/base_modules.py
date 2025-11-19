import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    """多头注意力机制模块"""
    def __init__(self, hidden_size, num_heads, dropout_prob, is_cross_attention=False, encoder_hidden_size=None):
        super().__init__()
        assert hidden_size % num_heads == 0, "hidden_size 必须能被 num_heads 整除"
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = self.head_dim ** -0.5
        self.query = nn.Linear(hidden_size, hidden_size)
        key_in_dim   = encoder_hidden_size if is_cross_attention else hidden_size
        value_in_dim = encoder_hidden_size if is_cross_attention else hidden_size
        self.key   = nn.Linear(key_in_dim, hidden_size)
        self.value = nn.Linear(value_in_dim, hidden_size)
        self.out = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, query, key, value, attention_mask=None):
        B, N, C = query.shape; Bk, S, Ck = key.shape
        assert B == Bk, "batch size mismatch"
        q = self.query(query).view(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.key(key).view(B, S, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.value(value).view(B, S, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        if attention_mask is not None: attn_weights = attn_weights + attention_mask
        attn_probs = F.softmax(attn_weights, dim=-1)
        attn_probs = self.dropout(attn_probs)
        out = torch.matmul(attn_probs, v).permute(0, 2, 1, 3).contiguous().view(B, N, C)
        return self.out(out)


class FeedForward(nn.Module):
    """前馈网络模块"""
    def __init__(self, hidden_size, intermediate_size, dropout_prob):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, intermediate_size)
        self.gelu = nn.GELU()
        self.fc2 = nn.Linear(intermediate_size, hidden_size)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x):
        return self.dropout(self.fc2(self.gelu(self.fc1(x))))


class DualQFormerLayer(nn.Module):
    """双向Q-Former层，包含自注意力、交叉注意力和前馈网络"""
    def __init__(self, hidden_size, num_heads, dropout_prob, intermediate_size, encoder_hidden_size):
        super().__init__()
        self.self_attn   = MultiHeadAttention(hidden_size, num_heads, dropout_prob, is_cross_attention=False)
        self.norm1       = nn.LayerNorm(hidden_size)
        self.cross_attn  = MultiHeadAttention(hidden_size, num_heads, dropout_prob, is_cross_attention=True, encoder_hidden_size=encoder_hidden_size)
        self.norm_cross  = nn.LayerNorm(hidden_size)
        self.ffn         = FeedForward(hidden_size, intermediate_size, dropout_prob)
        self.norm2       = nn.LayerNorm(hidden_size)

    def forward(self, joint_embeds, num_query_tokens, encoder_hidden_states, encoder_attention_mask, self_attn_mask):
        sa_out = self.self_attn(joint_embeds, joint_embeds, joint_embeds, attention_mask=self_attn_mask)
        joint  = self.norm1(joint_embeds + sa_out)
        q = joint[:, :num_query_tokens, :]
        ca_out = self.cross_attn(q, encoder_hidden_states, encoder_hidden_states, attention_mask=encoder_attention_mask)
        q = self.norm_cross(q + ca_out)
        joint = torch.cat([q, joint[:, num_query_tokens:, :]], dim=1)
        ffn_out = self.ffn(joint)
        joint   = self.norm2(joint + ffn_out)
        return joint


class DualQFormerEncoder(nn.Module):
    """Q-Former编码器，由多个DualQFormerLayer组成"""
    def __init__(self, num_layers, hidden_size, num_heads, dropout_prob, intermediate_size, encoder_hidden_size):
        super().__init__()
        self.layers = nn.ModuleList([
            DualQFormerLayer(hidden_size, num_heads, dropout_prob, intermediate_size, encoder_hidden_size)
            for _ in range(num_layers)
        ])

    def forward(self, joint_embeds, num_query_tokens, encoder_hidden_states, encoder_attention_mask, self_attn_mask):
        x = joint_embeds
        for layer in self.layers:
            x = layer(x, num_query_tokens, encoder_hidden_states, encoder_attention_mask, self_attn_mask)
        return x
