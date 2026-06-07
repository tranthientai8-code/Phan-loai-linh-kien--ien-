"""
demo.py — Demo phân loại linh kiện điện tử
==========================================
Cách dùng:
  python demo.py                          # chọn ảnh qua hộp thoại
  python demo.py --image path/to/img.jpg  # chỉ định ảnh trực tiếp
  python demo.py --folder path/to/folder  # dự đoán cả thư mục

Yêu cầu:
  pip install torch torchvision pillow
"""

import json
import argparse
import pathlib
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image

# ─── CẤU HÌNH ────────────────────────────────────────────────
SCRIPT_DIR  = pathlib.Path(__file__).parent.resolve()
MODEL_PATH  = SCRIPT_DIR / 'best_model.pth'
NAMES_PATH  = SCRIPT_DIR / 'class_names.json'
IMG_SIZE    = 224
TOP_K       = 5

MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

# ─── LOAD MODEL ──────────────────────────────────────────────
def load_model(model_path, class_names):
    model = models.efficientnet_b3()
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(in_features, 512),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, len(class_names))
    )
    ckpt = torch.load(model_path, map_location='cpu')
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    return model

# ─── TRANSFORM ───────────────────────────────────────────────
tfm = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

# ─── DỰ ĐOÁN 1 ẢNH ──────────────────────────────────────────
def predict(model, class_names, image_path, top_k=TOP_K):
    img = Image.open(image_path).convert('RGB')
    inp = tfm(img).unsqueeze(0)
    with torch.no_grad():
        out  = model(inp)
        prob = F.softmax(out, dim=1)[0]
        top  = prob.topk(top_k)

    results = []
    for p, i in zip(top.values.tolist(), top.indices.tolist()):
        results.append({'class': class_names[i], 'confidence': p})
    return results, img

# ─── IN KẾT QUẢ ──────────────────────────────────────────────
def print_result(image_path, results):
    print(f"\n{'='*50}")
    print(f"  Ảnh : {pathlib.Path(image_path).name}")
    print(f"{'='*50}")
    for i, r in enumerate(results):
        bar = '█' * int(r['confidence'] * 30)
        mark = ' ← TOP 1' if i == 0 else ''
        print(f"  [{i+1}] {r['class']:<35} {r['confidence']*100:5.1f}%  {bar}{mark}")
    print()

# ─── HIỂN THỊ ẢNH + KẾT QUẢ (nếu có matplotlib) ────────────
def show_image(image_path, results, img):
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        fig.patch.set_facecolor('#0a0e1a')

        # Ảnh
        ax1.imshow(img)
        ax1.set_title(f"Dự đoán: {results[0]['class']}\n{results[0]['confidence']*100:.1f}%",
                      color='white', fontsize=13, fontweight='bold')
        ax1.axis('off')
        ax1.set_facecolor('#0a0e1a')

        # Bar chart
        classes = [r['class'] for r in results]
        confs   = [r['confidence']*100 for r in results]
        colors  = ['#00ff88', '#00d4ff', '#7b5ea7', '#4a6080', '#2a3a4a']

        bars = ax2.barh(classes[::-1], confs[::-1], color=colors[::-1], height=0.6)
        ax2.set_xlim(0, 100)
        ax2.set_xlabel('Confidence (%)', color='#c8d8e8')
        ax2.set_title('Top 5 Dự đoán', color='white', fontsize=13, fontweight='bold')
        ax2.set_facecolor('#0f1525')
        ax2.tick_params(colors='#c8d8e8')
        ax2.spines['bottom'].set_color('#1e3a5f')
        ax2.spines['left'].set_color('#1e3a5f')
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)

        for bar, conf in zip(bars, confs[::-1]):
            ax2.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                     f'{conf:.1f}%', va='center', color='#00d4ff', fontsize=10)

        plt.suptitle(f'⚡ Phân loại linh kiện điện tử',
                     color='white', fontsize=15, fontweight='bold', y=1.01)
        plt.tight_layout()
        plt.show()
    except ImportError:
        print("[INFO] Cài matplotlib để xem đồ thị: pip install matplotlib")

# ─── CHỌN FILE QUA HỘP THOẠI ────────────────────────────────
def pick_file():
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        path = filedialog.askopenfilename(
            title='Chọn ảnh linh kiện',
            filetypes=[('Image files', '*.jpg *.jpeg *.png *.bmp *.webp'), ('All files', '*.*')]
        )
        root.destroy()
        return path if path else None
    except Exception:
        print("[WARN] Không thể mở hộp thoại. Dùng --image để chỉ định ảnh.")
        return None

# ─── MAIN ────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description='Demo phân loại linh kiện điện tử')
    parser.add_argument('--image',  type=str, default=None, help='Đường dẫn ảnh')
    parser.add_argument('--folder', type=str, default=None, help='Thư mục chứa nhiều ảnh')
    parser.add_argument('--no-plot', action='store_true', help='Không hiển thị đồ thị')
    args = parser.parse_args()

    # Kiểm tra file model
    if not MODEL_PATH.exists():
        print(f"❌ Không tìm thấy model: {MODEL_PATH}")
        print("   Đặt file best_model.pth cùng thư mục với demo.py")
        sys.exit(1)

    if not NAMES_PATH.exists():
        print(f"❌ Không tìm thấy: {NAMES_PATH}")
        print("   Đặt file class_names.json cùng thư mục với demo.py")
        sys.exit(1)

    # Load class names & model
    with open(NAMES_PATH, encoding='utf-8') as f:
        class_names = json.load(f)

    print(f"\n⚡ Đang load model EfficientNet-B3 ({len(class_names)} classes)...")
    model = load_model(MODEL_PATH, class_names)
    print(f"✅ Model ready!\n")

    # Xác định ảnh cần dự đoán
    if args.folder:
        # Dự đoán cả thư mục
        folder = pathlib.Path(args.folder)
        images = list(folder.glob('*.jpg')) + list(folder.glob('*.jpeg')) + \
                 list(folder.glob('*.png')) + list(folder.glob('*.webp'))
        if not images:
            print(f"❌ Không tìm thấy ảnh trong: {folder}")
            sys.exit(1)
        print(f"📁 Tìm thấy {len(images)} ảnh trong {folder}\n")
        for img_path in images:
            try:
                results, img = predict(model, class_names, img_path)
                print_result(img_path, results)
            except Exception as e:
                print(f"  [ERR] {img_path.name}: {e}")

    else:
        # Dự đoán 1 ảnh
        image_path = args.image
        if not image_path:
            image_path = pick_file()
        if not image_path:
            print("❌ Chưa chọn ảnh. Dùng: python demo.py --image path/to/image.jpg")
            sys.exit(1)

        if not pathlib.Path(image_path).exists():
            print(f"❌ Không tìm thấy file: {image_path}")
            sys.exit(1)

        results, img = predict(model, class_names, image_path)
        print_result(image_path, results)

        if not args.no_plot:
            show_image(image_path, results, img)


if __name__ == '__main__':
    main()