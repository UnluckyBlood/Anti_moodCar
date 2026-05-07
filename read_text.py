import tkinter as tk
from tkinter import filedialog
import cv2
from PIL import Image, ImageTk
from ultralytics import YOLO
import torch
import torch.nn as nn
from torchvision import transforms
import pystray
from PIL import Image as PILImage

# ────────── Конфиг ──────────
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
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1), nn.ReLU(True),
            nn.MaxPool2d(2,2),
            nn.Conv2d(64,128,3,padding=1), nn.ReLU(True),
            nn.MaxPool2d(2,2),
            nn.Conv2d(128,256,3,padding=1), nn.BatchNorm2d(256), nn.ReLU(True),
            nn.Conv2d(256,256,3,padding=1), nn.ReLU(True),
            nn.MaxPool2d((2,1),(2,1)),
            nn.Conv2d(256,512,3,padding=1), nn.BatchNorm2d(512), nn.ReLU(True),
            nn.Conv2d(512,512,3,padding=1), nn.ReLU(True),
            nn.MaxPool2d((2,1),(2,1))
        )
        self.rnn = nn.LSTM(512*2,256,bidirectional=True,num_layers=2,batch_first=True)
        self.classifier = nn.Linear(512,num_classes)

    def forward(self,x):
        x=self.cnn(x)
        b,c,h,w=x.size()
        x=x.reshape(b,c*h,w).permute(0,2,1)
        x,_=self.rnn(x)
        x=self.classifier(x)
        x=nn.functional.log_softmax(x,dim=2)
        return x.permute(1,0,2)

# ────────── OCR ──────────
class CRNNRecognizer:
    def __init__(self, path, alphabet):
        self.device=Config.DEVICE
        self.int_to_char={i+1:c for i,c in enumerate(alphabet)}
        self.int_to_char[0]=''

        self.model=CRNN(len(alphabet)+1).to(self.device)
        self.model.load_state_dict(torch.load(path,map_location=self.device))
        self.model.eval()

        self.transform=transforms.Compose([
            transforms.ToPILImage(),
            transforms.Grayscale(),
            transforms.Resize((32,128)),
            transforms.ToTensor(),
            transforms.Normalize([0.5],[0.5])
        ])

    @torch.no_grad()
    def recognize(self,img):
        img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        x=self.transform(img).unsqueeze(0).to(self.device)

        preds=self.model(x).permute(1,0,2)
        _,idx=torch.max(preds,2)
        idx=idx.view(-1).cpu().numpy()

        res=[]; prev=0
        for i in idx:
            if i!=0 and i!=prev:
                res.append(self.int_to_char.get(int(i),''))
            prev=i
        return ''.join(res)

# ────────── GUI ──────────
class ANPR_GUI:
    def __init__(self,root):
        self.root=root
        self.root.geometry("910x605")
        self.root.overrideredirect(True)

        print("Загрузка моделей...")
        self.detector=YOLO(Config.YOLO_MODEL_PATH)
        self.ocr=CRNNRecognizer(Config.OCR_MODEL_PATH,Config.OCR_ALPHABET)
        print("Готово!")

        self.canvas=tk.Canvas(root,width=910,height=605,highlightthickness=0)
        self.canvas.pack()

        # ────────── Загрузка картинок ──────────
        self.bg=ImageTk.PhotoImage(Image.open("bg_main.png"))
        self.top_bar=ImageTk.PhotoImage(Image.open("top_bar.png"))

        self.btn_upload=ImageTk.PhotoImage(Image.open("btn_upload.png"))
        self.btn_upload_h=ImageTk.PhotoImage(Image.open("btn_upload_hover.png"))

        self.btn_manual=ImageTk.PhotoImage(Image.open("btn_manual.png"))
        self.btn_manual_h=ImageTk.PhotoImage(Image.open("btn_manual_hover.png"))

        self.btn_close=ImageTk.PhotoImage(Image.open("btn_close.png"))
        self.btn_close_h=ImageTk.PhotoImage(Image.open("btn_close_hover.png"))

        self.btn_min=ImageTk.PhotoImage(Image.open("btn_min.png"))
        self.btn_min_h=ImageTk.PhotoImage(Image.open("btn_min_hover.png"))

        # ────────── Рисуем ──────────
        self.canvas.create_image(0,0,anchor=tk.NW,image=self.bg)
        self.top_bar_id=self.canvas.create_image(0,0,anchor=tk.NW,image=self.top_bar)

        self.upload_id=self.canvas.create_image(455,257,image=self.btn_upload)
        self.manual_id=self.canvas.create_image(455,545,image=self.btn_manual)

        self.close_id=self.canvas.create_image(899,12,image=self.btn_close)
        self.min_id=self.canvas.create_image(850,12,image=self.btn_min)

        # Поле для ручного ввода (изначально скрыто)
        self.manual_entry = tk.Entry(self.root, font=("Arial", 14), justify="center")
        self.manual_entry_window = self.canvas.create_window(455, 575, window=self.manual_entry, state="hidden", width=200)
        self.manual_number = tk.StringVar()   # Переменная для хранения номера
        self.manual_entry.config(textvariable=self.manual_number)

        # Идентификаторы для загруженной картинки и крестика (будут созданы позже)
        self.result_image_id = None
        self.clear_btn_id = None

        # ────────── Hover ──────────
        self.add_hover(self.upload_id,self.btn_upload,self.btn_upload_h)
        self.add_hover(self.manual_id,self.btn_manual,self.btn_manual_h)
        self.add_hover(self.close_id,self.btn_close,self.btn_close_h)
        self.add_hover(self.min_id,self.btn_min,self.btn_min_h)

        # ────────── Клики ──────────
        self.canvas.tag_bind(self.upload_id,"<Button-1>",self.load_image)
        self.canvas.tag_bind(self.manual_id,"<Button-1>",self.toggle_manual_entry)
        self.canvas.tag_bind(self.close_id,"<Button-1>",lambda e:self.root.destroy())
        self.canvas.tag_bind(self.min_id,"<Button-1>",self.minimize_to_tray)

        # ────────── Перетаскивание ──────────
        self.canvas.tag_bind(self.top_bar_id,"<ButtonPress-1>",self.start_move)
        self.canvas.tag_bind(self.top_bar_id,"<B1-Motion>",self.on_move)

        self.offset_x=0
        self.offset_y=0

        # ────────── Трей ──────────
        self.tray_icon = None
        # Загружаем иконку для трея (должен быть файл icon.ico или png)
        try:
            tray_img = PILImage.open("icon.ico")   # Путь к иконке
        except:
            tray_img = PILImage.new("RGB", (64,64), "blue")  # Заглушка
        self.tray_menu = pystray.Menu(
            pystray.MenuItem("Открыть", self.show_window),
            pystray.MenuItem("Выход", self.quit_app)
        )
        self.tray_icon = pystray.Icon("ANPR", tray_img, "ANPR", self.tray_menu)

        self.root.protocol('WM_DELETE_WINDOW', self.quit_app)  # чтобы при закрытии окна всё корректно завершалось

    # ────────── Hover функция ──────────
    def add_hover(self, item, normal, hover):
        def on_enter(e):
            self.canvas.itemconfig(item,image=hover)
        def on_leave(e):
            self.canvas.itemconfig(item,image=normal)
        self.canvas.tag_bind(item,"<Enter>",on_enter)
        self.canvas.tag_bind(item,"<Leave>",on_leave)

    # ────────── Перетаскивание ──────────
    def start_move(self,event):
        self.offset_x=event.x
        self.offset_y=event.y

    def on_move(self,event):
        x=self.root.winfo_pointerx()-self.offset_x
        y=self.root.winfo_pointery()-self.offset_y
        self.root.geometry(f"+{x}+{y}")

    # ────────── Загрузка изображения ──────────
    def load_image(self,event=None):
        path=filedialog.askopenfilename()
        if not path: return

        img=cv2.imread(path)
        if img is None: return

        self.detect(img)

    def detect(self,img):
        res=self.detector(img,conf=0.3)
        detected_text = ""
        for r in res:
            for b in r.boxes:
                x1,y1,x2,y2=map(int,b.xyxy[0])
                crop=img[y1:y2,x1:x2]
                text=self.ocr.recognize(crop)
                detected_text = text   # берём последний (или можно сохранить все)
                cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
                cv2.putText(img,text,(x1,y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0),2)

        self.show_result(img)
        if detected_text:
            self.manual_number.set(detected_text)   # запишем в переменную

    def show_result(self,img):
        self.canvas.delete(self.upload_id)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img).resize((self.btn_upload.width(), self.btn_upload.height()))
        self.result_img = ImageTk.PhotoImage(img)

        self.result_image_id = self.canvas.create_image(455, 257, image=self.result_img)

        w = self.btn_upload.width()//2
        h = self.btn_upload.height()//2
        cross_x = 455 + w - 15
        cross_y = 257 - h + 15
        self.clear_btn_id = self.canvas.create_text(cross_x, cross_y, text="✕",
                                                    font=("Arial", 16, "bold"), fill="red",
                                                    activefill="darkred")
        self.canvas.tag_bind(self.clear_btn_id, "<Button-1>", self.clear_image)

    def clear_image(self, event=None):
        if self.result_image_id:
            self.canvas.delete(self.result_image_id)
        if self.clear_btn_id:
            self.canvas.delete(self.clear_btn_id)
        self.result_image_id = None
        self.clear_btn_id = None

        self.upload_id = self.canvas.create_image(455,257, image=self.btn_upload)
        self.add_hover(self.upload_id, self.btn_upload, self.btn_upload_h)
        self.canvas.tag_bind(self.upload_id, "<Button-1>", self.load_image)
        self.manual_number.set("")

    def toggle_manual_entry(self, event=None):
        current_state = self.canvas.itemcget(self.manual_entry_window, "state")
        if current_state == "hidden":
            self.canvas.itemconfig(self.manual_entry_window, state="normal")
            self.manual_entry.focus_set()
        else:
            self.canvas.itemconfig(self.manual_entry_window, state="hidden")
            self.manual_number.set("")   # очищаем при скрытии? Можно оставить

    def minimize_to_tray(self, event=None):
        self.root.withdraw()   # скрыть окно
        if self.tray_icon:
            import threading
            if not hasattr(self, 'tray_thread') or not self.tray_thread.is_alive():
                self.tray_thread = threading.Thread(target=self.tray_icon.run, daemon=True)
                self.tray_thread.start()

    def show_window(self):
        self.root.deiconify()     # показать окно
        if self.tray_icon:
            self.tray_icon.stop()  # остановить иконку (будет перезапущена при следующем сворачивании)
        self.tray_thread = None

    def quit_app(self):
        if self.tray_icon:
            self.tray_icon.stop()
        self.root.destroy()

if __name__=="__main__":
    root=tk.Tk()
    app=ANPR_GUI(root)
    root.mainloop()