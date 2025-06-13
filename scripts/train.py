import argparse
import torch
from torch.utils.data import DataLoader, random_split
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from configs.gpt_config import GPTConfig
from models.gpt_model import GPT
from models.dataset import MyDataset

def train_epoch(model, loader, optimizer, scheduler, device, epoch, log_interval=100):
    """
    执行一个训练轮：
    - 遍历 DataLoader，前向、反向、优化、学习率调度
    - 定期打印损失
    """
    model.train()
    total_loss = 0.0
    for batch_idx, (x, y) in enumerate(loader):
        x, y = x.to(device), y.to(device)
        logits, loss = model(x, targets=y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        if batch_idx % log_interval == 0:
            print(f"Epoch {epoch} Batch {batch_idx} Loss {loss.item():.4f}")
    return total_loss / len(loader)

def eval_epoch(model, loader, device):
    """
    在验证集上计算平均损失
    """
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            _, loss = model(x, targets=y)
            val_loss += loss.item()
    return val_loss / len(loader)

def main():
    parser = argparse.ArgumentParser(description="Train GPT from scratch")
    parser.add_argument('--data_path',      type=str, required=True,  help="输入 JSONL 数据文件路径")
    parser.add_argument('--block_size',     type=int, default=512,     help="最大序列长度")
    parser.add_argument('--batch_size',     type=int, default=12,      help="训练批大小")
    parser.add_argument('--n_layer',        type=int, default=6,       help="Transformer 层数")
    parser.add_argument('--n_head',         type=int, default=12,      help="注意力头数")
    parser.add_argument('--n_embd',         type=int, default=768,     help="Embedding 大小")
    parser.add_argument('--max_steps',      type=int, default=1000,    help="调度 T_max")
    parser.add_argument('--lr',             type=float, default=3e-4,  help="初始学习率")
    parser.add_argument('--out_dir',        type=str, required=True,  help="模型保存目录")
    args = parser.parse_args()

    # 设备配置
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 加载数据集并切分训练/验证
    dataset = MyDataset(args.data_path, block_size=args.block_size)
    train_size = int(len(dataset) * 0.9)
    val_size   = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size)

    # 构建模型
    config = GPTConfig(
        block_size=args.block_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd
    )
    model = GPT(config).to(device)

    # 优化器与学习率调度
    optimizer = AdamW(model.parameters(), lr=args.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.max_steps)

    # 训练与验证循环
    for epoch in range(1, 3):
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, device, epoch)
        val_loss   = eval_epoch(model, val_loader, device)
        print(f"Epoch {epoch} Train Loss: {train_loss:.4f} Val Loss: {val_loss:.4f}")

        # 保存 checkpoint
        ckpt = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'val_loss': val_loss
        }
        torch.save(ckpt, f"{args.out_dir}/model_step_{epoch}.pt")

if __name__ == "__main__":
    main()
