import tkinter as tk
from tkinter import filedialog
import cv2
import numpy as np
from PIL import Image, ImageTk
from ultralytics import YOLO
import torch
import torch.nn as nn
from torchvision import transforms

# ────────── Конфигурация ──────────
class Config:
    YOLO_MODEL_PATH = "best.pt"
    OCR_MODEL_PATH = "crnn_ocr_model.pth"
    OCR_IMG_HEIGHT = 32
    OCR_IMG_WIDTH = 128
    OCR_ALPHABET = '0123456789ABCEHKMOPTXY'
    DEVICE = torch.device("cpu")

# ────────── CRNN ──────────
class CRNN(nn.Module):
    def __init__(self, num_classes):
        super(CRNN, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1), nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.ReLU(True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.ReLU(True),
            nn.MaxPool2d((2, 1), (2, 1)),
            nn.Conv2d(256, 512, kernel_size=3, padding=1), nn.BatchNorm2d(512), nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.ReLU(True),
            nn.MaxPool2d((2, 1), (2, 1))
        )
        self.rnn = nn.LSTM(512 * 2, 256, bidirectional=True, num_layers=2, batch_first=True)
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.cnn(x)                          # [batch, 512, 2, W]
        batch, ch, h, w = x.size()
        x = x.reshape(batch, ch * h, w)          # [batch, 1024, W]
        x = x.permute(0, 2, 1)                   # [batch, W, 1024]
        x, _ = self.rnn(x)                       # [batch, W, 512]
        x = self.classifier(x)                   # [batch, W, num_classes]
        x = nn.functional.log_softmax(x, dim=2)
        return x.permute(1, 0, 2)                # [W, batch, num_classes]

# ────────── Обёртка распознавателя ──────────
class CRNNRecognizer:
    def __init__(self, model_path, alphabet):
        self.device = Config.DEVICE
        self.alphabet = alphabet
        # Карты для CTC: 0 = blank, остальные 1..N
        self.int_to_char = {i + 1: ch for i, ch in enumerate(alphabet)}
        self.int_to_char[0] = ''
        self.num_classes = len(alphabet) + 1

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Grayscale(),
            transforms.Resize((Config.OCR_IMG_HEIGHT, Config.OCR_IMG_WIDTH)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

        # Загрузка модели
        self.model = CRNN(self.num_classes).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

    @torch.no_grad()
    def recognize(self, plate_img_bgr):
        # BGR → RGB → PIL → тензор
        plate_rgb = cv2.cvtColor(plate_img_bgr, cv2.COLOR_BGR2RGB)
        inp = self.transform(plate_rgb).unsqueeze(0).to(self.device)

        preds = self.model(inp)                     # [W, 1, num_classes]
        preds = preds.permute(1, 0, 2)              # [1, W, num_classes]
        _, max_idx = torch.max(preds, 2)
        indices = max_idx.view(-1).cpu().numpy()

        # Жадное декодирование CTC
        decoded = []
        prev = 0
        for idx in indices:
            idx = int(idx)
            if idx != 0 and idx != prev:
                decoded.append(self.int_to_char.get(idx, ''))
            prev = idx
        return ''.join(decoded)

# ────────── GUI ──────────
class ANPR_GUI:
    def __init__(self, root):
        self.root = root
        self.root.title("АвтоМудень")
        self.root.geometry("910x605")

        print("Загружаем YOLOv8...")
        self.detector = YOLO(Config.YOLO_MODEL_PATH)
        print("Загружаем CRNN...")
        self.ocr = CRNNRecognizer(Config.OCR_MODEL_PATH, Config.OCR_ALPHABET)
        print("Готов к работе!")

        self.original_image = None
        self.annotated_image = None
        self.tk_image = None

        control = tk.Frame(root)
        control.pack(side=tk.TOP, fill=tk.X, pady=5)
        tk.Button(control, text="Загрузить фото", command=self.load_image, width=15).pack(side=tk.LEFT, padx=10)
        self.btn_detect = tk.Button(control, text="Распознать номера", command=self.detect_and_recognize,
                                    state=tk.DISABLED, width=18)
        self.btn_detect.pack(side=tk.LEFT, padx=10)
        tk.Button(control, text="Очистить", command=self.clear, width=10).pack(side=tk.LEFT, padx=10)

        self.canvas = tk.Canvas(root, bg='gray', width=900, height=450)
        self.canvas.pack(pady=10)
        self.text_output = tk.Text(root, height=6, width=100)
        self.text_output.pack(pady=5)

    def load_image(self):
        path = filedialog.askopenfilename(filetypes=[("Изображения", "*.jpg *.jpeg *.png")])
        if not path: return
        self.original_image = cv2.imread(path)
        if self.original_image is None:
            self.text_output.insert(tk.END, "Ошибка загрузки\n")
            return
        self.display_image(self.original_image)
        self.btn_detect.config(state=tk.NORMAL)
        self.text_output.delete(1.0, tk.END)
        self.text_output.insert(tk.END, "Фото загружено.\n")

    def display_image(self, img_bgr):
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        cw = self.canvas.winfo_width() or 900
        ch = self.canvas.winfo_height() or 450
        pil_img.thumbnail((cw-20, ch-20), Image.Resampling.LANCZOS)
        self.tk_image = ImageTk.PhotoImage(pil_img)
        self.canvas.delete("all")
        self.canvas.create_image(cw//2, ch//2, image=self.tk_image, anchor=tk.CENTER)

    @staticmethod
    def nms_boxes(boxes, iou_thresh=0.4):
        if not boxes: return []
        boxes = sorted(boxes, key=lambda b: b[4], reverse=True)
        keep = []
        while boxes:
            best = boxes.pop(0)
            keep.append(best)
            boxes = [b for b in boxes if ANPR_GUI.iou(best[:4], b[:4]) < iou_thresh]
        return keep

    @staticmethod
    def iou(a, b):
        xA, yA = max(a[0], b[0]), max(a[1], b[1])
        xB, yB = min(a[2], b[2]), min(a[3], b[3])
        inter = max(0, xB-xA) * max(0, yB-yA)
        area_a = (a[2]-a[0])*(a[3]-a[1])
        area_b = (b[2]-b[0])*(b[3]-b[1])
        return inter / (area_a + area_b - inter + 1e-6)

    def detect_and_recognize(self):
        if self.original_image is None: return
        results = self.detector(self.original_image, conf=0.3, iou=0.5,
                                imgsz=1280, agnostic_nms=True, verbose=False)
        raw = []
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                raw.append([x1, y1, x2, y2, conf])
        boxes = self.nms_boxes(raw)
        if not boxes:
            self.text_output.delete(1.0, tk.END)
            self.text_output.insert(tk.END, "Номера не найдены.\n")
            self.display_image(self.original_image)
            return
        annotated = self.original_image.copy()
        self.text_output.delete(1.0, tk.END)
        h, w = self.original_image.shape[:2]
        for (x1, y1, x2, y2, conf) in boxes:
            margin = max(2, int((x2-x1)*0.05))
            x1c, y1c = max(0, x1-margin), max(0, y1-margin)
            x2c, y2c = min(w, x2+margin), min(h, y2+margin)
            crop = self.original_image[y1c:y2c, x1c:x2c]
            if crop.size == 0: continue
            text = self.ocr.recognize(crop)
            if not text: text = "?"
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.putText(annotated, f"{text} ({conf:.2f})", (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
            self.text_output.insert(tk.END, f"Номер: {text}  (увер.: {conf:.2f})\n")
        self.annotated_image = annotated
        self.display_image(annotated)

    def clear(self):
        self.original_image = None
        self.annotated_image = None
        self.canvas.delete("all")
        self.text_output.delete(1.0, tk.END)
        self.btn_detect.config(state=tk.DISABLED)

if __name__ == "__main__":
    root = tk.Tk()
    app = ANPR_GUI(root)
    root.mainloop()