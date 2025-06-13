import argparse
import torch
import tiktoken

from configs.gpt_config import GPTConfig
from models.gpt_model import GPT

def main():
    parser = argparse.ArgumentParser(description="Chat with trained GPT model")
    parser.add_argument('--checkpoint',    type=str, required=True, help="模型 checkpoint 文件路径")
    parser.add_argument('--block_size',    type=int, default=512,   help="最大序列长度（与训练一致）")
    parser.add_argument('--max_new_tokens',type=int, default=50,    help="生成的最大新 tokens 数")
    args = parser.parse_args()

    # 设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 初始化模型结构并加载权重
    config = GPTConfig(block_size=args.block_size)
    model = GPT(config).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    # 初始化编码器
    enc = tiktoken.get_encoding("gpt2")

    print("开始对话（输入 q 或 quit 退出）")
    while True:
        prompt = input("你：")
        if prompt.lower() in ("q", "quit", "exit"):
            break

        # 编码输入
        prompt_ids = enc.encode(prompt)
        idx = torch.tensor([prompt_ids], dtype=torch.long, device=device)

        # 生成后续 tokens
        with torch.no_grad():
            out = model.generate(idx, max_new_tokens=args.max_new_tokens)

        # 只取生成部分
        response_ids = out[0].tolist()[len(prompt_ids):]
        response = enc.decode(response_ids)
        print("模型：", response)

if __name__ == "__main__":
    main()
