import torch
import torch.nn.functional as F
import math
from torch import nn

class SingleHeadAttention(nn.Module):
    """
    单头自注意力模块：
    - 计算 Q, K, V
    - 计算注意力分数并做下三角掩码
    - softmax、dropout 后加权 V
    """
    def __init__(self, config):
        super().__init__()
        # 线性变换：Q、K、V
        self.query = nn.Linear(config.n_embd, config.head_size)
        self.key   = nn.Linear(config.n_embd, config.head_size)
        self.value = nn.Linear(config.n_embd, config.head_size)
        # 下三角掩码，形状 (block_size, block_size)
        self.register_buffer(
            'attention_mask',
            torch.tril(torch.ones(config.block_size, config.block_size))
        )
        self.dropout = nn.Dropout(config.dropout)
        self.head_size = config.head_size

    def forward(self, x):
        # x: (B, T, n_embd)
        B, T, _ = x.size()
        # 线性映射
        q = self.query(x)   # (B, T, head_size)
        k = self.key(x)     # (B, T, head_size)
        v = self.value(x)   # (B, T, head_size)
        # 计算注意力分数
        scores = q @ k.transpose(-2, -1)  # (B, T, T)
        # 掩码 & 缩放
        scores = scores.masked_fill(
            self.attention_mask[:T, :T] == 0,
            float('-inf')
        ) / math.sqrt(self.head_size)
        # softmax & dropout
        weights = F.softmax(scores, dim=-1)  # (B, T, T)
        weights = self.dropout(weights)
        # 加权求和值
        out = weights @ v  # (B, T, head_size)
        return out

class MultiHeadAttention(nn.Module):
    """
    多头自注意力模块：
    - 并行若干个 SingleHeadAttention
    - 拼接后再做线性投射和 dropout
    """
    def __init__(self, config):
        super().__init__()
        # 创建 n_head 个单头注意力
        self.heads = nn.ModuleList([
            SingleHeadAttention(config)
            for _ in range(config.n_head)
        ])
        # 投射回原始 embedding 大小
        self.proj = nn.Linear(config.head_size * config.n_head, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        # 并行处理每个头
        head_outputs = [h(x) for h in self.heads]  # List of (B, T, head_size)
        # 拼接维度
        concat = torch.cat(head_outputs, dim=-1)   # (B, T, head_size * n_head)
        # 投射回 n_embd
        out = self.proj(concat)                    # (B, T, n_embd)
        out = self.dropout(out)
        return out
