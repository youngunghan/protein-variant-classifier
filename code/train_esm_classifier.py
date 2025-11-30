import os
import argparse
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader, DistributedSampler, RandomSampler
from transformers import AutoTokenizer, EsmModel
from torch.nn.parallel import DistributedDataParallel as DDP

# --- Configuration Defaults ---
MODEL_NAME = "facebook/esm2_t33_650M_UR50D"

# --- Dataset ---
class VariantDataset(Dataset):
    def __init__(self, data, tokenizer, max_len):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        wt_seq, mut_seq, label = self.data[idx]
        
        wt_encoded = self.tokenizer(wt_seq, return_tensors='pt', padding='max_length', truncation=True, max_length=self.max_len)
        mut_encoded = self.tokenizer(mut_seq, return_tensors='pt', padding='max_length', truncation=True, max_length=self.max_len)
        
        return {
            'wt_input_ids': wt_encoded['input_ids'].squeeze(0),
            'wt_attention_mask': wt_encoded['attention_mask'].squeeze(0),
            'mut_input_ids': mut_encoded['input_ids'].squeeze(0),
            'mut_attention_mask': mut_encoded['attention_mask'].squeeze(0),
            'label': torch.tensor(label, dtype=torch.long)
        }

# --- Model ---
class ESM2VariantClassifier(nn.Module):
    def __init__(self, model_name=MODEL_NAME):
        super().__init__()
        self.esm = EsmModel.from_pretrained(model_name)
        
        # Backbone Freeze
        for param in self.esm.parameters():
            param.requires_grad = False
            
        hidden_size = self.esm.config.hidden_size
        
        # Classification Head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 3, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 2)
        )

    def forward(self, wt_input_ids, wt_attention_mask, mut_input_ids, mut_attention_mask):
        wt_outputs = self.esm(input_ids=wt_input_ids, attention_mask=wt_attention_mask)
        mut_outputs = self.esm(input_ids=mut_input_ids, attention_mask=mut_attention_mask)
        
        wt_cls = wt_outputs.last_hidden_state[:, 0, :]
        mut_cls = mut_outputs.last_hidden_state[:, 0, :]
        
        diff = mut_cls - wt_cls
        combined_features = torch.cat((wt_cls, mut_cls, diff), dim=1)
        
        logits = self.classifier(combined_features)
        return logits

# --- Training ---
def train(args):
    # Setup Device & Distributed
    is_distributed = args.local_rank != -1
    
    if is_distributed:
        dist.init_process_group(args.backend)
        local_rank = args.local_rank
        if args.backend == "nccl":
            torch.cuda.set_device(local_rank)
            device = torch.device("cuda", local_rank)
        else:
            device = torch.device("cpu")
        print(f"[Rank {local_rank}] Distributed training initialized with backend {args.backend}.")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Local training on {device}.")

    # Load Model
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = ESM2VariantClassifier(MODEL_NAME).to(device)
    
    if is_distributed:
        if args.backend == "nccl":
            model = DDP(model, device_ids=[local_rank], output_device=local_rank)
        else:
            model = DDP(model)

    # Mock Data (Replace with real data loading)
    if is_distributed and dist.get_rank() == 0:
        print("Generating mock data...")
    elif not is_distributed:
        print("Generating mock data...")
        
    mock_seq = "M" * 50
    # 9:1 Imbalance
    mock_data = [(mock_seq, mock_seq, 0)] * 90 + [(mock_seq, mock_seq, 1)] * 10
    
    dataset = VariantDataset(mock_data, tokenizer, args.max_len)
    
    if is_distributed:
        sampler = DistributedSampler(dataset)
    else:
        sampler = RandomSampler(dataset)
        
    dataloader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler)
    
    # Loss & Optimizer
    class_weights = torch.tensor([0.1, 0.9]).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    
    # Training Loop
    model.train()
    for epoch in range(args.epochs):
        if is_distributed:
            sampler.set_epoch(epoch)
            
        total_loss = 0
        for batch in dataloader:
            wt_ids = batch['wt_input_ids'].to(device)
            wt_mask = batch['wt_attention_mask'].to(device)
            mut_ids = batch['mut_input_ids'].to(device)
            mut_mask = batch['mut_attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            optimizer.zero_grad()
            outputs = model(wt_ids, wt_mask, mut_ids, mut_mask)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        # Print only on rank 0 or local
        if not is_distributed or dist.get_rank() == 0:
            avg_loss = total_loss / len(dataloader)
            print(f"Epoch {epoch+1}/{args.epochs}, Loss: {avg_loss:.4f}")

    if is_distributed:
        dist.destroy_process_group()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # DDP argument (automatically supplied by torchrun)
    parser.add_argument("--local_rank", type=int, default=-1, help="Local rank for DDP")
    parser.add_argument("--backend", type=str, default="nccl", help="Distributed backend (nccl for GPU, gloo for CPU)")
    
    # Hyperparameters
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size per GPU")
    parser.add_argument("--max_len", type=int, default=1024, help="Max sequence length")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    
    args = parser.parse_args()
    
    # Environment variable check for torchrun
    if "LOCAL_RANK" in os.environ:
        args.local_rank = int(os.environ["LOCAL_RANK"])
        
    train(args)
