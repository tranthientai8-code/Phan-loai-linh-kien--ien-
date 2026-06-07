# Phan-loai-linh-kien-dien
 Bộ phân loại linh kiện điện tử
Hệ thống phân loại hệ thống điện tử tự động sử dụng EfficiencyNet-B3 — nhận đầu vào hình ảnh và trả về độ tin cậy đính kèm dự kiến ​​​​top 5.

 Nội dung repo
.
├── best_model.pth                  # Checkpoint model tốt nhất (EfficientNet-B3)
├── electronic_classifier.pth       # Checkpoint model từ lần train gần nhất
├── electronic_resnet18_weights.pth # Checkpoint ResNet-18 (thử nghiệm)
├── class_names.json                # Danh sách 36 class linh kiện
├── demo.py                         # Script chạy demo dự đoán
└── train_model.py                  # Script huấn luyện model

🔧 Yêu cầu cài đặt
đậppip install torch torchvision pillow
pip install matplotlib   # Tuỳ chọn — để hiển thị đồ thị

GPU: Nếu có GPU, PyTorch sẽ tự động sử dụng. Nếu không, CPU vẫn hoạt động nhưng sẽ chậm hơn khi đào tạo. Khuyến nghị sử dụng Google Colab cho công việc huấn luyện.


🚀 Hướng dẫn sử dụng
Chạy demo dự đoán
đập# Chọn ảnh qua hộp thoại (GUI)
python demo.py

# Chỉ định ảnh trực tiếp
python demo.py --image path/to/image.jpg

# Dự đoán cả thư mục ảnh
python demo.py --folder path/to/folder/

# Không hiển thị đồ thị (chỉ in kết quả ra terminal)
python demo.py --image path/to/image.jpg --no-plot
Ví dụ đầu ra:
==================================================
  Ảnh : transistor_01.jpg
==================================================
  [1] junction-transistor                  87.3%  ██████████████████████████ ← TOP 1
  [2] PNP-transistor                        8.1%  ██
  [3] semiconductor-diode                   2.4%  
  [4] transistor                            1.6%  
  [5] semi-conductor                        0.6%
Use model in code Python
Pythonimport torch, json
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

 Mô hình huấn luyện lại
chuẩn bị dữ liệu
Sắp xếp ảnh theo thư mục cấu hình ImageFoldercủa PyTorch:
images/
├── transistor/
│   ├── img001.jpg
│   └── img002.jpg
├── LED/
│   ├── img001.jpg
│   └── ...
└── microchip/
    └── ...
Tập lệnh tự động tìm dữ liệu thư mục theo thứ tự ưu tiên: images/→ data/→ dataset/→ cùng thư mục với tập lệnh.
chạy huấn luyện
đậppython train_model.py
Quá trình đào tạo gồm 2 giai đoạn:
Giai đoạnMô tảEpochs mặc địnhTốc độ học tậpGiai đoạn 1 — Khởi độngĐóng băng xương sống, chỉ đào tạo đầu phân loại51e-3Giai đoạn 2 — Tinh chỉnhMở toàn bộ mô hình, tinh chỉnh từ đầu đến cuốitối đa 305e-5

Dừng sớm tự động dừng nếu độ chính xác xác thực không cải thiện sau 7 kỷ nguyên liên tiếp
Model tốt nhất (val acc cao nhất) được lưu vàoelectronic_classifier.pth
Tên lớp được lưu vàoclass_names.json


 Danh sách lớp 36
Tụ điện bypassTụ điện phânMạch vi tích hợpDẪN ĐẾNTransistor PNPphần ứngbộ suy giảmcầu chì hộpkẹp-dây dẫnrơle điệnsợitản nhiệtcuộn cảm ứngdây cáp nốibóng bán dẫn nốimạch đènbộ giới hạn-cắtbộ dao động cục bộchip nhớvi mạchbộ vi xử lýbộ ghép kênhĂng-ten đa hướngbộ chia điện thếchiết ápmáy phát xungrơlebiến trởchất bán dẫnđiốt bán dẫnđường vòngvan điện từbộ ổn địnhmáy biến áp hạ ápmáy biến áp tăng ápbóng bán dẫn

 Kiến trúc mẫu
EfficientNet-B3 (pretrained ImageNet)
└── Classifier head (tuỳ chỉnh):
    ├── Dropout(0.5)
    ├── Linear(in_features → 512)
    ├── ReLU
    ├── Dropout(0.3)
    └── Linear(512 → 36 classes)
Tăng cường dữ liệu khi huấn luyện:

Cắt ngẫu nhiên, lật ngang/dọc, hiệu ứng rung màu, xoay ±30°
Ảnh xám ngẫu nhiên (5%), Xóa/Cắt ngẫu nhiên (20%)
Chuẩn hóa theo giá trị trung bình/độ lệch chuẩn của ImageNet.

Hàm mất mát: CrossEntropyLoss với làm mịn nhãn Trình 0.1
tối ưu hóa: AdamW, giảm trọng lượng 1e-2
Bộ lập lịch: Cosine Annealing LR

📁 Điểm kiểm tra dạng định dạng
Python{
    'model_state_dict': ...,   # Trọng số model
    'class_names':      [...], # Danh sách tên class
    'num_classes':      36,    # Số lượng class
    'val_acc':          0.xx,  # Val accuracy tốt nhất
}
