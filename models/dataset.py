import json
import torch
from torch.utils.data import Dataset
import tiktoken

class MyDataset(Dataset):
    """
    自定义 Dataset：
    - 读取 JSONL，每行 JSON 包含 'text'
    - 使用 tiktoken GPT2 编码
    - 拼接并按 block_size 切分为训练样本
    """
    def __init__(self, path, block_size=512, max_lines=1000):
        # 初始化编码器
        self.enc = tiktoken.get_encoding("gpt2")
        self.block_size = block_size
        # EOS token ID
        self.eos_token = self.enc.encode(
            "<|endoftext|>",
            allowed_special={"<|endoftext|>"}
        )[0]

        # 读取前 max_lines 行文本
        raw_texts = []
        with open(path, 'r', encoding='utf8') as f:
            for i, line in enumerate(f):
                if i >= max_lines:
                    break
                try:
                    txt = json.loads(line)['text']
                    raw_texts.append(txt)
                except:
                    continue

        # 编码所有文本并拼接
        all_tokens = []
        for txt in raw_texts:
            tokens = self.enc.encode(txt)
            all_tokens.extend(tokens + [self.eos_token])

        # 切分为块
        self.encoded_data = []
        for i in range(0, len(all_tokens), block_size):
            chunk = all_tokens[i:i+block_size+1]
            # 不足则填 EOS
            if len(chunk) < block_size + 1:
                chunk += [self.eos_token] * (block_size + 1 - len(chunk))
            self.encoded_data.append(chunk)

    def __len__(self):
        return len(self.encoded_data)

    def __getitem__(self, idx):
        chunk = self.encoded_data[idx]
        x = torch.tensor(chunk[:-1], dtype=torch.long)  # 输入
        y = torch.tensor(chunk[1:], dtype=torch.long)   # 预测目标
        return x, y
