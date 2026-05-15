"""
Training script for Transformer language model.

Author: Apala Pramanik
"""

import os
import sys
import time
import itertools
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.dataset import CharDataset
from src.model.transformer_model import TransformerLanguageModel
from src.model.attention import causal_mask


# ======================================================
# CONFIG
# ======================================================
SEQ_LEN      = 128
BATCH_SIZE   = 128
EMBED_DIM    = 256
NUM_HEADS    = 8
NUM_LAYERS   = 6
FF_DIM       = 1024
DROPOUT      = 0.1
WEIGHT_DECAY = 1e-5
PATIENCE     = 15
EPOCHS       = 200
LR           = 3e-4
GRAD_CLIP    = 1.0
TIME_LIMIT_S = 35 * 3600

DEVICE  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP = DEVICE.type == "cuda"

CHECKPOINT_DIR = "checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

JOB_START = time.time()


# ======================================================
# DATA
# ======================================================
with open("data/wikitext-2-raw/wiki.train.raw", "r", encoding="utf-8") as f:
    train_text = f.read()

with open("data/wikitext-2-raw/wiki.valid.raw", "r", encoding="utf-8") as f:
    val_text = f.read()

train_dataset = CharDataset(train_text, SEQ_LEN, stride=SEQ_LEN)
val_dataset   = CharDataset(val_text,   SEQ_LEN, vocab=train_dataset.stoi, stride=SEQ_LEN)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  drop_last=True, num_workers=1, pin_memory=True)
val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False, drop_last=True, num_workers=1, pin_memory=True)

VOCAB_SIZE    = train_dataset.vocab_size
TOTAL_BATCHES = len(train_loader)
MID_BATCH     = TOTAL_BATCHES // 2
PRINT_EVERY   = max(1, TOTAL_BATCHES // 5)

print(f"Device: {DEVICE}  |  Vocab: {VOCAB_SIZE}  |  Batches/epoch: {TOTAL_BATCHES}")


# ======================================================
# MODEL
# ======================================================
model = TransformerLanguageModel(
    vocab_size=VOCAB_SIZE,
    embed_dim=EMBED_DIM,
    num_heads=NUM_HEADS,
    num_layers=NUM_LAYERS,
    ff_hidden_dim=FF_DIM,
    dropout=DROPOUT,
).to(DEVICE)

optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
criterion = nn.CrossEntropyLoss()

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.5, patience=2, min_lr=1e-5,
)

scaler = torch.amp.GradScaler("cuda", enabled=USE_AMP)


# ======================================================
# CHECKPOINTING
# ======================================================
def save_checkpoint(epoch, batch_idx, total_loss, total_correct, total_tokens,
                    train_losses, val_losses, train_accuracies, val_accuracies,
                    best_val_loss, patience_counter, suffix=""):
    path = os.path.join(CHECKPOINT_DIR, f"epoch_{epoch}{suffix}.pt")
    torch.save({
        "epoch":            epoch,
        "batch_idx":        batch_idx,
        "model_state":      model.state_dict(),
        "optimizer_state":  optimizer.state_dict(),
        "scaler_state":     scaler.state_dict(),
        "scheduler_state":  scheduler.state_dict(),
        "total_loss":       total_loss,
        "total_correct":    total_correct,
        "total_tokens":     total_tokens,
        "train_losses":     train_losses,
        "val_losses":       val_losses,
        "train_accuracies": train_accuracies,
        "val_accuracies":   val_accuracies,
        "best_val_loss":    best_val_loss,
        "patience_counter": patience_counter,
    }, path)
    print(f"  Checkpoint saved: epoch_{epoch}{suffix}.pt")


def find_latest_checkpoint():
    for epoch in range(EPOCHS, 0, -1):
        for suffix in ["_end", "_time", "_mid", ""]:
            path = os.path.join(CHECKPOINT_DIR, f"epoch_{epoch}{suffix}.pt")
            if os.path.exists(path):
                ckpt = torch.load(path, map_location="cpu", weights_only=False)
                return path, epoch, ckpt.get("batch_idx", None)
    return None, 0, None


# ======================================================
# RESUME
# ======================================================
train_losses, val_losses = [], []
train_accuracies, val_accuracies = [], []
start_epoch, start_batch = 1, 0
best_val_loss    = float("inf")
patience_counter = 0

ckpt_path, ckpt_epoch, ckpt_batch = find_latest_checkpoint()
if ckpt_path is not None:
    print(f"Resuming from {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    optimizer.load_state_dict(ckpt["optimizer_state"])
    scaler.load_state_dict(ckpt["scaler_state"])
    scheduler.load_state_dict(ckpt["scheduler_state"])
    train_losses     = ckpt["train_losses"]
    val_losses       = ckpt["val_losses"]
    train_accuracies = ckpt.get("train_accuracies", [])
    val_accuracies   = ckpt.get("val_accuracies",   [])
    best_val_loss    = ckpt.get("best_val_loss",    float("inf"))
    patience_counter = ckpt.get("patience_counter", 0)
    if ckpt_batch is None:
        start_epoch = ckpt_epoch + 1
        start_batch = 0
    else:
        start_epoch = ckpt_epoch
        start_batch = ckpt_batch + 1
    print(f"  Resuming at epoch {start_epoch}, batch {start_batch}")
else:
    ckpt = None


# ======================================================
# VALIDATION
# ======================================================
@torch.no_grad()
def evaluate(model, dataloader):
    model.eval()
    total_loss    = 0.0
    total_correct = 0
    total_tokens  = 0
    for x, y in dataloader:
        x, y   = x.to(DEVICE), y.to(DEVICE)
        mask   = causal_mask(x.size(1), DEVICE)
        logits = model(x, mask)
        total_loss    += criterion(logits.view(-1, VOCAB_SIZE), y.view(-1)).item()
        preds          = logits.view(-1, VOCAB_SIZE).argmax(-1)
        total_correct += (preds == y.view(-1)).sum().item()
        total_tokens  += y.numel()
    model.train()
    return total_loss / len(dataloader), total_correct / total_tokens


# ======================================================
# TEXT GENERATION
# ======================================================
@torch.no_grad()
def generate_text(model, dataset, start_text="The market", length=200, temperature=1.0):
    model.eval()
    indices = [dataset.stoi.get(c, 0) for c in start_text]
    x = torch.tensor(indices, dtype=torch.long).unsqueeze(0).to(DEVICE)
    for _ in range(length):
        x_in  = x[:, -SEQ_LEN:]
        mask  = causal_mask(x_in.size(1), DEVICE)
        logits = model(x_in, mask)
        next_logits = logits[0, -1] / temperature
        probs = torch.softmax(next_logits, dim=-1)
        next_idx = torch.multinomial(probs, 1).item()
        x = torch.cat([x, torch.tensor([[next_idx]], device=DEVICE)], dim=1)
    print("\n===== GENERATED TEXT =====\n")
    print("".join(dataset.itos[i] for i in x[0].tolist()))
    print("\n==========================\n")
    model.train()


# ======================================================
# TRAINING LOOP
# ======================================================
model.train()

for epoch in range(start_epoch, EPOCHS + 1):
    epoch_start_batch = start_batch if epoch == start_epoch else 0
    total_loss    = (ckpt["total_loss"]           if (ckpt is not None and epoch == start_epoch and epoch_start_batch > 0) else 0.0)
    total_correct = (ckpt.get("total_correct", 0) if (ckpt is not None and epoch == start_epoch and epoch_start_batch > 0) else 0)
    total_tokens  = (ckpt.get("total_tokens",  0) if (ckpt is not None and epoch == start_epoch and epoch_start_batch > 0) else 0)

    torch.manual_seed(42 + epoch)
    train_iter = iter(train_loader)

    if epoch_start_batch > 0:
        print(f"  Skipping {epoch_start_batch} already-processed batches...")
        for _ in itertools.islice(train_iter, epoch_start_batch):
            pass

    for batch_offset, (x, y) in enumerate(train_iter):
        batch_idx = epoch_start_batch + batch_offset
        x, y  = x.to(DEVICE), y.to(DEVICE)
        mask  = causal_mask(x.size(1), DEVICE)
        optimizer.zero_grad()

        if USE_AMP:
            with torch.amp.autocast("cuda"):
                logits = model(x, mask)
                loss   = criterion(logits.view(-1, VOCAB_SIZE), y.view(-1))
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(x, mask)
            loss   = criterion(logits.view(-1, VOCAB_SIZE), y.view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()

        total_loss    += loss.item()
        with torch.no_grad():
            preds          = logits.detach().view(-1, VOCAB_SIZE).argmax(-1)
            total_correct += (preds == y.view(-1)).sum().item()
        total_tokens  += y.numel()

        if batch_idx % PRINT_EVERY == 0:
            print(f"Epoch {epoch} | Batch {batch_idx}/{TOTAL_BATCHES} | Loss {loss.item():.4f}")

        if batch_idx == MID_BATCH:
            save_checkpoint(epoch, batch_idx, total_loss, total_correct, total_tokens,
                            train_losses, val_losses, train_accuracies, val_accuracies,
                            best_val_loss, patience_counter, suffix="_mid")

        if batch_offset % 100 == 0 and (time.time() - JOB_START) > TIME_LIMIT_S:
            save_checkpoint(epoch, batch_idx, total_loss, total_correct, total_tokens,
                            train_losses, val_losses, train_accuracies, val_accuracies,
                            best_val_loss, patience_counter, suffix="_time")
            print("Time limit reached. Exiting cleanly.")
            sys.exit(0)

    train_loss = total_loss / TOTAL_BATCHES
    train_acc  = total_correct / total_tokens if total_tokens > 0 else 0.0
    val_loss, val_acc = evaluate(model, val_loader)

    train_losses.append(train_loss)
    val_losses.append(val_loss)
    train_accuracies.append(train_acc)
    val_accuracies.append(val_acc)

    if val_loss < best_val_loss:
        best_val_loss    = val_loss
        patience_counter = 0
        save_checkpoint(epoch, None, total_loss, total_correct, total_tokens,
                        train_losses, val_losses, train_accuracies, val_accuracies,
                        best_val_loss, patience_counter, suffix="_best")
    else:
        patience_counter += 1

    print(f"\nEpoch {epoch} DONE | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f} | Patience: {patience_counter}/{PATIENCE}\n")

    scheduler.step(val_loss)
    save_checkpoint(epoch, None, total_loss, total_correct, total_tokens,
                    train_losses, val_losses, train_accuracies, val_accuracies,
                    best_val_loss, patience_counter, suffix="_end")

    generate_text(model, train_dataset, start_text="The market", length=200)

    if patience_counter >= PATIENCE:
        print(f"Early stopping triggered after {epoch} epochs.")
        open(os.path.join(CHECKPOINT_DIR, "DONE"), "w").close()
        break


open(os.path.join(CHECKPOINT_DIR, "DONE"), "w").close()


# ======================================================
# PLOT
# ======================================================
epochs_x = list(range(1, len(train_losses) + 1))

plt.figure()
plt.plot(epochs_x, train_losses, marker="o", label="Train Loss")
plt.plot(epochs_x, val_losses,   marker="s", label="Val Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.ylim(0, 8)
plt.title("Transformer — Training & Validation Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("loss_curve_trans.png", dpi=600)
plt.close()

plt.figure()
plt.plot(epochs_x, train_accuracies, marker="o", label="Train Accuracy")
plt.plot(epochs_x, val_accuracies,   marker="s", label="Val Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.ylim(0, 1)
plt.title("Transformer — Training & Validation Accuracy")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("accuracy_curve_trans.png", dpi=600)
plt.close()
