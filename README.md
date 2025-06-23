# MyGPT
一个纯 PyTorch（无外部大模型库）手写版 GPT 项目，包含模型、数据处理、训练脚本和对话推理示例。
A pure PyTorch, from-scratch GPT project (no external large-model libraries), including the model, data processing, training scripts, and dialogue inference examples.


```text
                 _______           _______  _______ _________
                (       )|\     /|(  ____ \(  ____ )\__   __/
                | () () |( \   / )| (    \/| (    )|   ) (   
                | || || | \ (_) / | |      | (____)|   | |   
                | |(_)| |  \   /  | | ____ |  _____)   | |   
                | |   | |   ) (   | | \_  )| (         | |   
                | )   ( |   | |   | (___) || )         | |   
                |/     \|   \_/   (_______)|/          )_(   
 ```                         


安装依赖
```bash
pip install -r requirements.txt
```

## 目录结构
```text
mygpt/
├── README.md
├── requirements.txt
├── configs/
│   └── gpt_config.py         # GPTConfig 数据类，管理超参
├── models/
│   ├── attention.py          # SingleHeadAttention, MultiHeadAttention
│   ├── block.py              # Transformer Block 和 FeedForward
│   ├── gpt_model.py          # GPT 主模型：Embedding, Blocks, LM Head, generate()
│   └── dataset.py            # MyDataset：JSONL 加载、编码、切分逻辑
├── data/
│   └── mobvoi_seq_monkey_general_open_corpus.jsonl
├── scripts/
│   ├── train.py              # 训练脚本，包含 argparse, 训练/验证循环、checkpoint 保存
│   └── chat.py               # 推理对话脚本，加载 checkpoint，REPL 生成回复
└── checkpoints/              # 存放训练输出的 .pt 模型文件

```




- **configs/gpt_config.py**  
  定义 `GPTConfig` 数据类，集中管理超参数。

- **models/gpt.py**  
  包含 `GPT`, `Block`, `MultiHeadAttention`, `FeedForward` 等模块化实现。

- **models/attention.py**  
  包含 `SingleHeadAttention`, `MultiHeadAttention`等模块化实现。

- **models/block.py**  
  包含 `FeedForward`, `Block`等模块化实现。

- **models/dataset.py**  
  包含 `MyDataset` 类：负责加载 JSONL 文本、编码、切分为训练样本。

- **data/mobvoi_seq_monkey_general_open_corpus.jsonly**  
  包含训练数据

- **scripts/train.py**  
  训练流程入口：加载数据、初始化模型 & 优化器 & 学习率调度器、执行训练 & 验证并保存 checkpoint。

- **scripts/chat.py**  
  推理流程示例：加载指定 checkpoint、输入 prompt、生成回复。

- **checkpoints/**  
  存放训练过程中保存的 `.pt` 文件。

训练示例
```bash
python scripts/train.py \
  --data_path    data/mobvoi_seq_monkey_general_open_corpus.jsonl \
  --block_size   512 \
  --batch_size   12 \
  --n_layer      6 \
  --n_head       12 \
  --n_embd       768 \
  --max_steps    1000 \
  --lr           3e-4 \
  --out_dir      checkpoints/

```

对话示例
```bash
python scripts/chat.py \
  --checkpoint     checkpoints/model_step_1000.pt \
  --block_size     512 \
  --max_new_tokens 50

```





