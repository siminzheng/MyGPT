import torch
import torch.nn.functional as F
from torch import nn
from configs.gpt_config import GPTConfig
from models.block import Block

class GPT(nn.Module):
    """
    GPT 主模型：
    - Token Embedding + Positional Embedding
    - 多个 Transformer Block
    - 最终 LayerNorm + LM Head
    - generate() 方法用于自回归生成
    """
    def __init__(self, config: GPTConfig):
        super().__init__()
        # token 与位置 embedding
        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)
        self.pos_embedding   = nn.Embedding(config.block_size, config.n_embd)
        # Transformer Blocks 序列
        self.blocks = nn.Sequential(
            *[Block(config) for _ in range(config.n_layer)]
        )
        self.ln_final = nn.LayerNorm(config.n_embd)
        self.lm_head  = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # 权重初始化
        self.apply(self._init_weights)

    def _init_weights(self, module):
        # 线性层与 Embedding 使用正态分布初始化
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if hasattr(module, 'bias') and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, idx, targets=None):
        """
        :param idx: (B, T) token IDs
        :param targets: (B, T) 预测目标，可选
        :return: logits (B, T, V), loss (可选)
        """
        B, T = idx.size()
        # embedding
        token_emb = self.token_embedding(idx)             # (B, T, n_embd)
        pos_ids   = torch.arange(T, device=idx.device)    # (T,)
        pos_emb   = self.pos_embedding(pos_ids)[None, :, :]  # (1, T, n_embd)
        x = token_emb + pos_emb                           # 广播相加

        # 经过所有 Block
        x = self.blocks(x)                                # (B, T, n_embd)
        x = self.ln_final(x)                              # (B, T, n_embd)
        logits = self.lm_head(x)                          # (B, T, vocab_size)

        # 如果给定 targets，则计算交叉熵损失
        loss = None
        if targets is not None:
            logits_flat  = logits.view(-1, logits.size(-1))  # (B*T, V)
            targets_flat = targets.view(-1)                  # (B*T,)
            loss = F.cross_entropy(logits_flat, targets_flat)
        return logits, loss

    def generate(self, idx, max_new_tokens):
        """
        自回归生成：
        按步迭代调用 forward，采样下一个 token。
        """
        for _ in range(max_new_tokens):
            B, T = idx.size()
            # 如果输入超长，截取最后 block_size 长度
            if T > self.token_embedding.num_embeddings:
                idx_cond = idx[:, -self.token_embedding.num_embeddings:]
            else:
                idx_cond = idx
            logits, _ = self(idx_cond)        # (B, T, V)
            next_logits = logits[:, -1, :]    # (B, V)
            probs = F.softmax(next_logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)  # (B, 1)
            idx = torch.cat([idx, next_id], dim=1)             # 加到序列末
        return idx
