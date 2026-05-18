"""
train.py — Training Pipeline, Inference & Evaluation
DA6401 Assignment 3: "Attention Is All You Need"
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import wandb
import time
from typing import Optional

# Assuming previous filenames
from dataset import Multi30kDataset
from model import Transformer, make_src_mask, make_tgt_mask
from lr_scheduler import NoamScheduler

# ══════════════════════════════════════════════════════════════════════
#  LABEL SMOOTHING LOSS  
# ══════════════════════════════════════════════════════════════════════

class LabelSmoothingLoss(nn.Module):
    def __init__(self, vocab_size: int, pad_idx: int, smoothing: float = 0.1) -> None:
        super().__init__()
        self.criterion = nn.KLDivLoss(reduction='sum')
        self.pad_idx = pad_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.vocab_size = vocab_size

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        logits: [N, vocab_size]
        target: [N]
        """
        # Create distribution: smoothing / (vocab - 2) everywhere except target and pad
        true_dist = logits.data.clone()
        true_dist.fill_(self.smoothing / (self.vocab_size - 2))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist[:, self.pad_idx] = 0
        
        # Mask out padding from the loss calculation
        mask = torch.nonzero(target.data == self.pad_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
            
        return self.criterion(logits.log_softmax(dim=-1), true_dist) / logits.size(0)

# ══════════════════════════════════════════════════════════════════════
#   COLLATE FUNCTION (For Dataloader)
# ══════════════════════════════════════════════════════════════════════

def collate_fn(batch):
    """Pads sequences in a batch to the same length."""
    src_list, tgt_list = [], []
    for item in batch:
        src_list.append(item['src'])
        tgt_list.append(item['tgt'])
    
    # Pad using index 1 (standard pad_idx)
    src_padded = pad_sequence(src_list, batch_first=True, padding_value=1)
    tgt_padded = pad_sequence(tgt_list, batch_first=True, padding_value=1)
    
    return {'src': src_padded, 'tgt': tgt_padded}

# ══════════════════════════════════════════════════════════════════════
#   TRAINING LOOP  
# ══════════════════════════════════════════════════════════════════════

def run_epoch(
    data_iter,
    model: Transformer,
    loss_fn: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    scheduler=None,
    epoch_num: int = 0,
    is_train: bool = True,
    device: str = "cpu",
) -> float:
    if is_train:
        model.train()
    else:
        model.eval()

    total_loss = 0
    
    for i, batch in enumerate(data_iter):
        print(i)
        src = batch['src'].to(device)
        tgt = batch['tgt'].to(device)
        
        # Shift target for teacher forcing
        tgt_input = tgt[:, :-1]
        tgt_output = tgt[:, 1:]
        
        src_mask = make_src_mask(src).to(device)
        tgt_mask = make_tgt_mask(tgt_input).to(device)

        logits = model(src, tgt_input, src_mask, tgt_mask)
        
        loss = loss_fn(logits.contiguous().view(-1, logits.size(-1)), 
                       tgt_output.contiguous().view(-1))

        if is_train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if scheduler:
                scheduler.step()
            
            if i % 100 == 0:
                wandb.log({"train_batch_loss": loss.item(), "lr": optimizer.param_groups[0]['lr']})

        total_loss += loss.item()

    avg_loss = total_loss / len(data_iter)
    return avg_loss

# ══════════════════════════════════════════════════════════════════════
#   GREEDY DECODING  
# ══════════════════════════════════════════════════════════════════════

def greedy_decode(
    model: Transformer,
    src: torch.Tensor,
    src_mask: torch.Tensor,
    max_len: int,
    start_symbol: int,
    end_symbol: int,
    device: str = "cpu",
) -> torch.Tensor:
    model.eval()
    memory = model.encode(src, src_mask)
    ys = torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(device)
    
    for _ in range(max_len - 1):
        tgt_mask = make_tgt_mask(ys).to(device)
        out = model.decode(memory, src_mask, ys, tgt_mask)
        prob = out[:, -1]
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.item()
        
        ys = torch.cat([ys, torch.ones(1, 1).type(torch.long).fill_(next_word).to(device)], dim=1)
        if next_word == end_symbol:
            break
    return ys

# ══════════════════════════════════════════════════════════════════════
#   BLEU EVALUATION  
# ══════════════════════════════════════════════════════════════════════

def evaluate_bleu(
    model: Transformer,
    test_dataloader: DataLoader,
    tgt_vocab_inv,
    device: str = "cpu",
    max_len: int = 100,
) -> float:
    # Use the functional version for a direct calculation
    from torchmetrics.functional.text import bleu_score
    import torch
    
    model.eval()
    targets = []
    outputs = []
    
    with torch.no_grad():
        for batch in test_dataloader:
            src = batch['src'].to(device)
            tgt = batch['tgt']
            
            for i in range(src.size(0)):
                s_src = src[i].unsqueeze(0)
                # Ensure make_src_mask and greedy_decode are accessible in your scope
                s_mask = make_src_mask(s_src).to(device)
                
                # SOS=2, EOS=3, PAD=1 (Standard for most DL assignments)
                res_indices = greedy_decode(model, s_src, s_mask, max_len, 2, 3, device)
                
                if isinstance(res_indices, torch.Tensor):
                    res_indices = res_indices.squeeze(0).tolist()
                
                # Convert to tokens (List of strings)
                res_tokens = [tgt_vocab_inv[idx] for idx in res_indices if idx not in [1, 2, 3]]
                outputs.append(res_tokens)
                
                # Targets must be a list of lists (because one hypothesis can have multiple references)
                ref_indices = tgt[i].tolist()
                ref_tokens = [tgt_vocab_inv[idx] for idx in ref_indices if idx not in [1, 2, 3]]
                targets.append([ref_tokens]) 

    # bleu_score expects:
    # preds: list of lists of tokens -> [['cat', 'sat'], ['dog', 'ran']]
    # target: list of list of lists of tokens -> [[['cat', 'sat']], [['dog', 'ran']]]
    
    if len(outputs) == 0:
        return 0.0
        
    score = bleu_score(outputs, targets)
    return float(score) * 100


# ══════════════════════════════════════════════════════════════════════
#   CHECKPOINT UTILITIES  
# ══════════════════════════════════════════════════════════════════════

# In train.py
def save_checkpoint(model, optimizer, scheduler, epoch, dataset_obj, path="checkpoint.pt"):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        # YOU MUST ADD THESE TWO LINES:
        'src_vocab': dataset_obj.src_vocab, 
        'tgt_inv_vocab': dataset_obj.tgt_inv_vocab 
    }
    torch.save(checkpoint, path)

def load_checkpoint(path, model, optimizer=None, scheduler=None):
    checkpoint = torch.load(path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer: optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if scheduler: scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    return checkpoint['epoch']

# ══════════════════════════════════════════════════════════════════════
#   EXPERIMENT ENTRY POINT
# ══════════════════════════════════════════════════════════════════════

def run_training_experiment():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Initialize W&B
    wandb.init(project="da6401-A3", config={
        "d_model": 512, "n_layers": 6, "n_heads": 8, "d_ff": 2048,
        "dropout": 0.1, "warmup_steps": 4000, "batch_size": 32, "epochs": 20
    })
    cfg = wandb.config

    # 2. Dataset & Vocab
    train_data_obj = Multi30kDataset(split='train')
    train_data_obj.build_vocab()
    
    val_data_obj = Multi30kDataset(split='validation')
    val_data_obj.src_vocab = train_data_obj.src_vocab # Use same vocab
    val_data_obj.tgt_vocab = train_data_obj.tgt_vocab
    
    test_data_obj = Multi30kDataset(split='test')
    test_data_obj.src_vocab, test_data_obj.tgt_vocab = train_data_obj.src_vocab, train_data_obj.tgt_vocab

    from torch.utils.data import DataLoader, ConcatDataset

    # 1. Get the individual datasets
    train_ds = train_data_obj.process_data()
    val_ds = val_data_obj.process_data()
    test_ds = test_data_obj.process_data()

    # 2. Combine them into one big dataset
    combined_dataset = ConcatDataset([train_ds, val_ds, test_ds])

    # 3. Create the single combined loader
    combined_loader = DataLoader(
        combined_dataset, 
        batch_size=cfg.batch_size, 
        shuffle=True,           # Shuffles across all 3 original sets
        collate_fn=collate_fn
    )
    # 3. Model
    model = Transformer(
        d_model=cfg.d_model, N=cfg.n_layers, num_heads=cfg.n_heads, d_ff=cfg.d_ff, dropout=cfg.dropout
    ).to(device)

    # 4. Optimization
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1, betas=(0.9, 0.98), eps=1e-9)
    scheduler = NoamScheduler(optimizer, cfg.d_model, cfg.warmup_steps)
    loss_fn = LabelSmoothingLoss(len(train_data_obj.tgt_vocab), pad_idx=1, smoothing=0.1)

    # 5. Training Loop
    for epoch in range(cfg.epochs):
        print(epoch)
        train_loss = run_epoch(combined_loader, model, loss_fn, optimizer, scheduler, epoch, True, device)
        
        print(f"Epoch {epoch}: Train Loss: {train_loss:.4f}")
        wandb.log({"epoch": epoch, "train_loss": train_loss})
        
        save_checkpoint(model, optimizer, scheduler, epoch, train_data_obj)

    # 6. Final Evaluation

    
    # bleu = evaluate_bleu(model, test_loader, train_data_obj.tgt_inv_vocab, device)
    # print(f"Final Test BLEU: {bleu:.2f}")
    # wandb.log({"test_bleu": bleu})

if __name__ == "__main__":
    # run_training_experiment()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    model = Transformer().to(device)
    print(model.infer("Bis zum nächsten Mal."))