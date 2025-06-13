import torch.nn as nn
from models.attention import MultiHeadAttention

class FeedForward(nn.Module):
    """
    前馈网络 (MLP)：
    Linear → GELU → Linear → Dropout
    """
    def __init__(self, config):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.dropout)
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """
    Transformer Block：
    - LayerNorm → MultiHeadAttention → 残差连接
    - LayerNorm → FeedForward → 残差连接
    """
    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.att = MultiHeadAttention(config)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.ffn = FeedForward(config)

    def forward(self, x):
        # 自注意力子层 + 残差
        x = x + self.att(self.ln1(x))
        # 前馈子层 + 残差
        x = x + self.ffn(self.ln2(x))
        return x
