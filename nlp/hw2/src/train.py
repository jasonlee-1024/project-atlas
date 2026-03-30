"""
Training script for GPT2ForSequenceClassification on 20 Newsgroups dataset.

This script trains a GPT-2 based classifier without using any HuggingFace libraries.
"""

import argparse
import json
import os
import time
from typing import List, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import wandb
from tqdm import tqdm

from gpt2 import GPT2Config, GPT2ForSequenceClassification


class NewsgroupsDataset(Dataset):
    def __init__(self, jsonl_path: str, max_length: int = 512):
        self.samples = []
        with open(jsonl_path, "r") as f:
            for line in f:
                data = json.loads(line)
                token_ids = data["token_ids"][:max_length]
                token_ids += [0] * (max_length - len(token_ids))
                self.samples.append({
                    "token_ids": token_ids,
                    "label": data["label"],
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        return (
            torch.tensor(sample["token_ids"], dtype=torch.long),
            torch.tensor(sample["label"], dtype=torch.long),
        )


def evaluate(model, dataloader, device):
    model.eval()
    total_correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            out = model(inputs)
            preds = out.logits.argmax(dim=-1)
            total_correct += (preds == targets).sum().item()
            total += len(targets)
    return total_correct / total


def train(args):
    # 1. Initialize wandb run with hyperparameters (lr, batch_size, epochs, max_length, etc.)
    wandb.init(
        project="gpt2-newsgroups",
        config={
            "lr": args.lr,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "max_length": args.max_length,
        }
    )

    # 2. Initialize device, model, optimizer, loss function
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = GPT2Config()
    model = GPT2ForSequenceClassification(config=config, lm_bin_path=args.lm_bin_path)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # 3. Load data: Dataset + DataLoader (train / val)
    train_data = NewsgroupsDataset(args.train_data, max_length=args.max_length)
    val_data = NewsgroupsDataset(args.val_data, max_length=args.max_length)
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True
    )
    validation_loader = torch.utils.data.DataLoader(
        val_data, batch_size=args.batch_size, shuffle=False
    )

    # 4. Training loop
    # for each epoch:
    #   for each batch:
    #     - forward
    #     - compute loss
    #     - backward
    #     - optimizer step
    #     - log step loss to wandb

    for epoch in tqdm(range(args.epochs)):
        model.train()

        train_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            out = model(inputs)
            loss = F.cross_entropy(out.logits, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            wandb.log({"train_loss": loss.item()})

        # 5. Evaluate on val_loader at the end of each epoch
        #    - log val_acc and avg train_loss to wandb
        val_acc = evaluate(model, validation_loader, device)
        avg_loss = train_loss / len(train_loader)
        wandb.log({"val_acc": val_acc, "avg_train_loss": avg_loss, "epoch": epoch + 1})
        print(f"Epoch {epoch+1} avg_loss={avg_loss:.4f} val_acc={val_acc:.4f}")

    # 6. Save checkpoint
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    torch.save(model.state_dict(), args.output_path)
    print(f"Saved checkpoint to {args.output_path}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-data", default="../data/20_newsgroups_train.jsonl")
    parser.add_argument("--val-data", default="../data/20_newsgroups_val.jsonl")
    parser.add_argument("--lm-bin-path", default="../checkpoints/gpt2_model.pth")
    parser.add_argument("--output-path", default="../checkpoints/classifier_model.pth")
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-5)
    args = parser.parse_args()

    train(args)
