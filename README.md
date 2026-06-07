# Phan-loai-linh-kien-dien
  Electronic Component Classifier

Hệ thống phân loại linh kiện điện tử tự động sử dụng **EfficientNet-B3** — nhận ảnh đầu vào và trả về top-5 dự đoán kèm độ tin cậy.

---

##  Nội dung repo

```
.
├── best_model.pth                  # Checkpoint model tốt nhất (EfficientNet-B3)
├── electronic_classifier.pth       # Checkpoint model từ lần train gần nhất
├── electronic_resnet18_weights.pth # Checkpoint ResNet-18 (thử nghiệm)
├── class_names.json                # Danh sách 36 class linh kiện
├── demo.py                         # Script chạy demo dự đoán
└── train_model.py                  # Script huấn luyện model
```

---

##  Yêu cầu cài đặt

```bash
pip install torch torchvision pillow
pip install matplotlib   # Tuỳ chọn — để hiển thị đồ thị
```

> **GPU:** Nếu có GPU, PyTorch sẽ tự động sử dụng. Nếu không, CPU vẫn hoạt động nhưng sẽ chậm hơn khi train. Khuyến nghị dùng [Google Colab](https://colab.research.google.com/) cho việc huấn luyện.

---

##  Hướng dẫn sử dụng

### Chạy demo dự đoán

```bash
# Chọn ảnh qua hộp thoại (GUI)
python demo.py

# Chỉ định ảnh trực tiếp
python demo.py --image path/to/image.jpg

# Dự đoán cả thư mục ảnh
python demo.py --folder path/to/folder/

# Không hiển thị đồ thị (chỉ in kết quả ra terminal)
python demo.py --image path/to/image.jpg --no-plot
```

**Ví dụ output:**

```
==================================================
  Ảnh : transistor_01.jpg
==================================================
  [1] junction-transistor                  87.3%  ██████████████████████████ ← TOP 1
  [2] PNP-transistor                        8.1%  ██
  [3] semiconductor-diode                   2.4%  
  [4] transistor                            1.6%  
  [5] semi-conductor                        0.6%  
```

### Sử dụng model trong code Python

```python
import torch, json
from torchvision import models, transforms
from PIL import Image

# Load model
ckpt  = torch.load('best_model.pth', map_location='cpu')
names = ckpt['class_names']

model = models.efficientnet_b3()
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

# Transform & predict
tfm = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

img = Image.open('your_image.jpg').convert('RGB')
with torch.no_grad():
    prob = torch.softmax(model(tfm(img).unsqueeze(0)), dim=1)[0]
    top5 = prob.topk(5)

for p, i in zip(top5.values, top5.indices):
    print(f'{names[i]}: {p*100:.1f}%')
```

---

##  Huấn luyện lại model

### Chuẩn bị dữ liệu

Sắp xếp ảnh theo cấu trúc thư mục `ImageFolder` của PyTorch:

```
images/
├── transistor/
│   ├── img001.jpg
│   └── img002.jpg
├── LED/
│   ├── img001.jpg
│   └── ...
└── microchip/
    └── ...
```

Script tự động tìm thư mục data theo thứ tự ưu tiên: `images/` → `data/` → `dataset/` → cùng thư mục với script.

### Chạy training

```bash
python train_model.py
```

**Quá trình training gồm 2 giai đoạn:**

| Giai đoạn | Mô tả | Epochs mặc định | Learning Rate |
|---|---|---|---|
| Phase 1 — Warm-up | Đóng băng backbone, chỉ train classifier head | 5 | `1e-3` |
| Phase 2 — Fine-tune | Mở toàn bộ model, fine-tune end-to-end | tối đa 30 | `5e-5` |

- Early stopping tự động dừng nếu validation accuracy không cải thiện sau **7 epoch liên tiếp**
- Model tốt nhất (val acc cao nhất) được lưu vào `electronic_classifier.pth`
- Class names được lưu vào `class_names.json`

---

##  Danh sách 36 class

| | | | |
|---|---|---|---|
| Bypass-capacitor | Electrolytic-capacitor | Integrated-micro-circuit | LED |
| PNP-transistor | armature | attenuator | cartridge-fuse |
| clip-lead | electric-relay | filament | heat-sink |
| induction-coil | jumper-cable | junction-transistor | light-circuit |
| limiter-clipper | local-oscillator | memory-chip | microchip |
| microprocessor | multiplexer | omni-directional-antenna | potential-divider |
| potentiometer | pulse-generator | relay | rheostat |
| semi-conductor | semiconductor-diode | shunt | solenoid |
| stabilizer | step-down-transformer | step-up-transformer | transistor |

---

##  Kiến trúc model

```
EfficientNet-B3 (pretrained ImageNet)
└── Classifier head (tuỳ chỉnh):
    ├── Dropout(0.5)
    ├── Linear(in_features → 512)
    ├── ReLU
    ├── Dropout(0.3)
    └── Linear(512 → 36 classes)
```

**Data augmentation khi train:**
- Random crop, horizontal/vertical flip, color jitter, rotation ±30°
- Random grayscale (5%), Random Erasing / Cutout (20%)
- Normalize theo ImageNet mean/std

**Loss:** CrossEntropyLoss với label smoothing `0.1`  
**Optimizer:** AdamW, weight decay `1e-2`  
**Scheduler:** Cosine Annealing LR

---

##  Định dạng checkpoint

```python
{
    'model_state_dict': ...,   # Trọng số model
    'class_names':      [...], # Danh sách tên class
    'num_classes':      36,    # Số lượng class
    'val_acc':          0.xx,  # Val accuracy tốt nhất
}
```
