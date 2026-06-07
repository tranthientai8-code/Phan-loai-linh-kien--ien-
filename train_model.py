"""
train_model.py — Phân loại linh kiện điện tử
============================================
Fix:
  - Tự động tìm đúng DATA_DIR (không hardcode './data')
  - Loại bỏ thư mục 'images' khỏi danh sách class
  - 2-phase training: warm-up → fine-tune toàn bộ
  - Tự động dùng GPU nếu có
  - Tốc độ nhanh hơn nhờ num_workers và pin_memory
"""

import os
import sys
import time
import pathlib
import json

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Subset
from torchvision import datasets, models, transforms

# ─── 1. CẤU HÌNH ──────────────────────────────────────────────────────────────

# Tự tìm thư mục data: ưu tiên thư mục 'images' bên cạnh script,
# hoặc thư mục hiện tại chứa các class-folder
SCRIPT_DIR = pathlib.Path(__file__).parent.resolve()

def find_data_dir():
    # Kiểm tra các vị trí phổ biến
    candidates = [
        SCRIPT_DIR / 'images',
        SCRIPT_DIR / 'data',
        SCRIPT_DIR / 'dataset',
        SCRIPT_DIR,               # class-folders ngay bên cạnh script
    ]
    for c in candidates:
        if c.is_dir():
            subdirs = [d for d in c.iterdir() if d.is_dir()]
            if len(subdirs) >= 5:  # Có ít nhất 5 class-folder
                return c
    return SCRIPT_DIR

DATA_DIR    = find_data_dir()
SAVE_PATH   = SCRIPT_DIR / 'electronic_classifier.pth'
JSON_PATH   = SCRIPT_DIR / 'class_names.json'

IMG_SIZE    = 224
BATCH_SIZE  = 32
SEED        = 42

# Số epoch cho từng giai đoạn
EPOCHS_WARMUP   = 5    # đóng băng backbone
EPOCHS_FINETUNE = 30   # fine-tune toàn bộ (có early stopping)

LR_WARMUP   = 1e-3
LR_FINETUNE = 5e-5     # Giảm mạnh để tránh overfit

PATIENCE = 7           # Early stopping: dừng nếu val không tăng sau 7 epoch

# Các thư mục cần loại bỏ (không phải class thật)
EXCLUDE_DIRS = {'images', 'data', 'dataset', '__pycache__', '.git'}

# ─── 2. DEVICE ─────────────────────────────────────────────────────────────────

torch.manual_seed(SEED)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[INFO] Device: {DEVICE}")
if DEVICE.type == 'cuda':
    print(f"[INFO] GPU: {torch.cuda.get_device_name(0)}")
else:
    print("[WARN] Không tìm thấy GPU — train sẽ chậm. Khuyến nghị dùng Google Colab.")

NUM_WORKERS = 0 if DEVICE.type == 'cuda' else 0
PIN_MEMORY  = (DEVICE.type == 'cuda')

# ─── 3. DATASET ────────────────────────────────────────────────────────────────

print(f"\n[INFO] DATA_DIR = {DATA_DIR}")

# Transforms
train_tfm = transforms.Compose([
    transforms.Resize((IMG_SIZE + 32, IMG_SIZE + 32)),
    transforms.RandomCrop(IMG_SIZE),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(p=0.3),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3, hue=0.1),
    transforms.RandomRotation(30),
    transforms.RandomGrayscale(p=0.05),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
    transforms.RandomErasing(p=0.2),   # Cutout augmentation
])

val_tfm = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

# Load dataset — dùng ImageFolder với is_valid_file để lọc class rác
full_dataset = datasets.ImageFolder(root=str(DATA_DIR))

# Lọc bỏ các thư mục không phải class thật
valid_classes = [c for c in full_dataset.classes if c not in EXCLUDE_DIRS]
valid_class_to_idx = {c: i for i, c in enumerate(valid_classes)}

# Lấy chỉ các sample thuộc class hợp lệ
valid_samples = [
    (path, valid_class_to_idx[full_dataset.classes[lbl]])
    for path, lbl in full_dataset.samples
    if full_dataset.classes[lbl] not in EXCLUDE_DIRS
]

NUM_CLASSES = len(valid_classes)
print(f"[INFO] Số class hợp lệ: {NUM_CLASSES}")
print(f"[INFO] Số ảnh hợp lệ  : {len(valid_samples)}")
print(f"[INFO] Danh sách class : {valid_classes}")

# Tạo dataset tuỳ chỉnh
class FilteredDataset(torch.utils.data.Dataset):
    def __init__(self, samples, transform=None):
        self.samples   = samples
        self.transform = transform
        self.loader    = datasets.folder.default_loader

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = self.loader(path)
        if self.transform:
            img = self.transform(img)
        return img, label

# Split 80 / 10 / 10
n_total = len(valid_samples)
n_val   = int(n_total * 0.10)
n_test  = int(n_total * 0.10)
n_train = n_total - n_val - n_test

gen = torch.Generator().manual_seed(SEED)
indices = torch.randperm(n_total, generator=gen).tolist()

train_idx = indices[:n_train]
val_idx   = indices[n_train:n_train + n_val]
test_idx  = indices[n_train + n_val:]

train_samples = [valid_samples[i] for i in train_idx]
val_samples   = [valid_samples[i] for i in val_idx]
test_samples  = [valid_samples[i] for i in test_idx]

train_ds = FilteredDataset(train_samples, transform=train_tfm)
val_ds   = FilteredDataset(val_samples,   transform=val_tfm)
test_ds  = FilteredDataset(test_samples,  transform=val_tfm)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)

print(f"[INFO] Train={len(train_ds)} | Val={len(val_ds)} | Test={len(test_ds)}")

# ─── 4. MODEL ──────────────────────────────────────────────────────────────────

weights = models.EfficientNet_B3_Weights.DEFAULT
model   = models.efficientnet_b3(weights=weights)
in_features = model.classifier[1].in_features
model.classifier = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(in_features, 512),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(512, NUM_CLASSES)
)
model = model.to(DEVICE)
print(f"[INFO] Model: EfficientNet-B3, output={NUM_CLASSES} classes")

criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

# ─── 5. HELPER FUNCTIONS ───────────────────────────────────────────────────────

def freeze_backbone(model):
    """Đóng băng tất cả layer trừ classifier."""
    for name, p in model.named_parameters():
        p.requires_grad = name.startswith('classifier.')

def unfreeze_all(model):
    """Mở toàn bộ layer."""
    for p in model.parameters():
        p.requires_grad = True

def count_trainable(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def train_one_epoch(model, loader, optimizer):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        out  = model(imgs)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * imgs.size(0)
        correct      += (out.argmax(1) == labels).sum().item()
        total        += imgs.size(0)
    return running_loss / total, correct / total

@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        out  = model(imgs)
        loss = criterion(out, labels)
        running_loss += loss.item() * imgs.size(0)
        correct      += (out.argmax(1) == labels).sum().item()
        total        += imgs.size(0)
    return running_loss / total, correct / total

# ─── 6. TRAINING ───────────────────────────────────────────────────────────────

best_val_acc = 0.0
history = []

def run_phase(phase_name, epochs, lr, freeze=False):
    global best_val_acc

    if freeze:
        freeze_backbone(model)
    else:
        unfreeze_all(model)

    trainable = count_trainable(model)
    print(f"\n{'='*60}")
    print(f"  {phase_name}  |  LR={lr}  |  Epochs={epochs}")
    print(f"  Params trainable: {trainable:,}")
    print(f"{'='*60}")

    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr, weight_decay=1e-2   # Tăng weight decay để giảm overfit
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=lr * 0.01
    )

    no_improve = 0   # Early stopping counter

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        tl, ta = train_one_epoch(model, train_loader, optimizer)
        vl, va = evaluate(model, val_loader)
        scheduler.step()
        elapsed = time.time() - t0

        flag = ''
        if va > best_val_acc:
            best_val_acc = va
            no_improve = 0
            torch.save({
                'model_state_dict': model.state_dict(),
                'class_names':      valid_classes,
                'num_classes':      NUM_CLASSES,
                'val_acc':          va,
            }, str(SAVE_PATH))
            flag = '  ✅ saved'
        else:
            no_improve += 1

        history.append({'phase': phase_name, 'epoch': epoch,
                         'train_loss': tl, 'train_acc': ta,
                         'val_loss': vl, 'val_acc': va})

        print(f"  Epoch {epoch:02d}/{epochs} | "
              f"Train loss={tl:.4f} acc={ta*100:.1f}% | "
              f"Val loss={vl:.4f} acc={va*100:.1f}% | "
              f"{elapsed:.0f}s{flag}")

        # Early stopping
        if not freeze and no_improve >= PATIENCE:
            print(f"\n[Early Stop] Val không cải thiện sau {PATIENCE} epoch — dừng sớm.")
            break

t_start = time.time()

# Phase 1: Warm-up (chỉ train FC head)
run_phase("Phase 1 — Warm-up (frozen backbone)",
          epochs=EPOCHS_WARMUP, lr=LR_WARMUP, freeze=True)

# Phase 2: Fine-tune toàn bộ
run_phase("Phase 2 — Fine-tuning (all layers)",
          epochs=EPOCHS_FINETUNE, lr=LR_FINETUNE, freeze=False)

total_time = time.time() - t_start
print(f"\n[SUCCESS] Hoàn thành trong: {total_time/60:.1f} phút")
print(f"[INFO] Best Val Accuracy: {best_val_acc*100:.2f}%")
print(f"[INFO] Model saved: {SAVE_PATH}")

# ─── 7. TEST SET EVALUATION ────────────────────────────────────────────────────

print("\n[INFO] Đánh giá trên Test Set...")
ckpt = torch.load(str(SAVE_PATH), map_location=DEVICE)
model.load_state_dict(ckpt['model_state_dict'])

test_loss, test_acc = evaluate(model, test_loader)
print(f"[RESULT] Test Loss={test_loss:.4f} | Test Accuracy={test_acc*100:.2f}%")

# ─── 8. LƯU CLASS NAMES ────────────────────────────────────────────────────────

with open(str(JSON_PATH), 'w', encoding='utf-8') as f:
    json.dump(valid_classes, f, ensure_ascii=False, indent=2)
print(f"[INFO] Class names saved: {JSON_PATH}")

# ─── 9. INFERENCE HÀM TIỆN ÍCH ─────────────────────────────────────────────────

print("\n--- Cách dùng model sau khi train ---")
print("""
import torch, json
from torchvision import models, transforms
from PIL import Image

ckpt   = torch.load('electronic_classifier.pth', map_location='cpu')
names  = ckpt['class_names']
model  = models.efficientnet_b3()
in_features = model.classifier[1].in_features
model.classifier = torch.nn.Sequential(
    torch.nn.Dropout(0.5),
    torch.nn.Linear(in_features, 512),
    torch.nn.ReLU(),
    torch.nn.Dropout(0.3),
    torch.nn.Linear(512, len(names))
)
model.load_state_dict(ckpt['model_state_dict'])
model.eval()

tfm = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])

img   = Image.open('your_image.jpg').convert('RGB')
with torch.no_grad():
    prob  = torch.softmax(model(tfm(img).unsqueeze(0)), 1)[0]
    top5  = prob.topk(5)
for p, i in zip(top5.values, top5.indices):
    print(f'{names[i]}: {p*100:.1f}%')
""")