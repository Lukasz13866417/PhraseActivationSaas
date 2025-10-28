from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import torch, time
from data_maker import *
from model import *
from augmentation import *

import torch
import torch.nn.functional as F

TARGET_T = 200  

def collate_fixed_time(batch):
    xs, ys = zip(*batch)  # xs: list of [1, n_mels, T_i]
    X = []
    for x in xs:
        x = x.contiguous()                   # be safe for stacking
        T = x.shape[-1]
        if T > TARGET_T:                     # center-crop
            start = (T - TARGET_T) // 2
            x = x[..., start:start + TARGET_T]
        elif T < TARGET_T:                   # right-pad with zeros
            pad = TARGET_T - T
            x = F.pad(x, (0, pad))
        X.append(x)
    X = torch.stack(X, dim=0)                # (B, 1, n_mels, TARGET_T)
    y = torch.tensor(ys, dtype=torch.long)
    return X, y

def train_one_epoch(model, loader, optimizer, scaler, device):
    model.train()
    loss_meter = 0.0
    for x, y in loader:
        x = x.to(device)                  
        y = y.float().to(device)         
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device.type, enabled=(device.type == "cuda")):
            logits, _ = model(x)
            loss = F.binary_cross_entropy_with_logits(logits, y)
        scaler.scale(loss).backward()
        scaler.step(optimizer); scaler.update()
        loss_meter += loss.item() * x.size(0)
    return loss_meter / len(loader.dataset)

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    loss_meter, probs, labels = 0.0, [], []
    for x, y in loader:
        x = x.to(device); y = y.float().to(device)
        logits, _ = model(x)
        loss = F.binary_cross_entropy_with_logits(logits, y)
        loss_meter += loss.item() * x.size(0)
        probs.append(torch.sigmoid(logits).cpu())
        labels.append(y.cpu())
    probs = torch.cat(probs); labels = torch.cat(labels)
    # simple metrics
    pred = (probs >= 0.5).int()
    acc = (pred == labels.int()).float().mean().item()
    return loss_meter/len(loader.dataset), acc, probs.numpy(), labels.numpy()


def eval_with_threshold(model, loader, device):
    model.eval()
    import torch.nn.functional as F
    loss_meter, probs, labels = 0.0, [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device); y = y.float().to(device)
            logits, _ = model(x)
            loss = F.binary_cross_entropy_with_logits(logits, y)
            loss_meter += loss.item() * x.size(0)
            probs.append(torch.sigmoid(logits).cpu())
            labels.append(y.cpu())
    probs = torch.cat(probs).numpy()
    labels = torch.cat(labels).numpy().astype(int)

    auc = roc_auc_score(labels, probs)
    fpr, tpr, thr = roc_curve(labels, probs)
    best = (tpr - fpr).argmax()
    th_opt = thr[best]

    pred = (probs >= th_opt).astype(int)
    acc = (pred == labels).mean()
    return loss_meter/len(loader.dataset), auc, th_opt, acc

# 1) Build data
df = make_df_from_db("db/tts.sqlite", "hello", limit_pos=1000, limit_neg=3000)                        
# split (super simple): last 10% as val
val_frac = 0.1
cut = int(len(df)*(1-val_frac))
df_train, df_val = train_test_split(
    df, test_size=0.1, stratify=df["label"], random_state=42
)

cfg = SpectrogramConfig(mean_db=-13.2, std_db=7.8)  
ds_train = KeywordDataset(df=df_train, cfg=cfg, augment=simple_augment)  
ds_val   = KeywordDataset(df=df_val,   cfg=cfg, augment=None)

train_loader = DataLoader(ds_train, batch_size=32, shuffle=True,
                          num_workers=16, pin_memory=True,
                          collate_fn=collate_fixed_time)
val_loader   = DataLoader(ds_val, batch_size=64, shuffle=False,
                          num_workers=16, pin_memory=True,
                          collate_fn=collate_fixed_time)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TinyCRNN(in_ch=1, n_mels=cfg.n_mels, hidden=128).to(device)
opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
scaler = torch.amp.GradScaler(device.type, enabled=(device.type == "cuda"))

best_val = float("inf"); best_path = "kw_model.pt"
for epoch in range(1, 2000):
    ds_train.set_epoch(epoch)            
    t0 = time.time()
    tr_loss = train_one_epoch(model, train_loader, opt, scaler, device)
    va_loss, va_auc, th_opt, va_acc_opt = eval_with_threshold(model, val_loader, device)
    print(f"val {va_loss:.4f} | AUC {va_auc:.3f} | thr* {th_opt:.3f} | acc@thr* {va_acc_opt:.3f}")
    if va_loss < best_val:
        best_val = va_loss
        torch.save({"model": model.state_dict(), "cfg": cfg.__dict__}, best_path)


ckpt = torch.load(best_path, map_location="cpu")
model.load_state_dict(ckpt["model"])
model.eval()