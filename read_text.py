import tkinter as tk
from tkinter import filedialog, messagebox
import cv2
import copy
import torch
import torch.nn as nn
from ultralytics import YOLO
from torchvision import transforms
from PIL import Image, ImageTk, ImageDraw
from PIL import Image as PILImage
import pystray
import threading

# =========================================================
# CONFIG
# =========================================================

class Config:
    YOLO_MODEL_PATH = "best.pt"
    OCR_MODEL_PATH   = "crnn_ocr_model.pth"
    OCR_ALPHABET     = '0123456789ABCEHKMOPTXY'
    DEVICE           = torch.device("cpu")

    USERS = {
        "admin": {
            "password": "admin123",
            "role": "moderator"
        },
        "user": {
            "password": "123",
            "role": "user"
        }
    }

    # Изначальная база
    TEMP_DB = {
        "A123BC": {
            "name": "Иван Иванов",
            "rating": 4.7,
            "votes": 12,
            "warning": "",
            "reviews": [
                {
                    "author": "Петр",
                    "text": "Очень спокойно ездит",
                    "approved": True,
                    "rating": 5,
                    "tags": ["polite", "signals"]
                }
            ]
        },
        "B777OO": {
            "name": "Сергей",
            "rating": 1.8,
            "votes": 4,
            "warning": "Опасное поведение на дороге",
            "reviews": [
                {
                    "author": "Макс",
                    "text": "Подрезает и сигналит",
                    "approved": False,
                    "rating": 1,
                    "tags": ["danger"]
                }
            ]
        }
    }

# =========================================================
# ТЕГИ (счетчики для иконок возле номера)
# =========================================================

RATING_TAGS = {
    "polite":   {"text": "Хорошее общение", "icon": "💬", "threshold": 3},
    "signals":  {"text": "Поворотники",     "icon": "🚘", "threshold": 3},
    "careful":  {"text": "Аккуратность",    "icon": "🛡", "threshold": 3},
    "helpful":  {"text": "Помощь",          "icon": "🤝", "threshold": 2},
    "parking":  {"text": "Парковка",        "icon": "🅿", "threshold": 3},
    "danger":   {"text": "Опасный",         "icon": "⚠", "threshold": 1}  # опасный показывается сразу
}

# =========================================================
# CRNN (свёрточная рекуррентная сеть)
# =========================================================

class CRNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1,64,3,padding=1), nn.ReLU(True),
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
        self.rnn = nn.LSTM(512*2, 256, bidirectional=True, num_layers=2, batch_first=True)
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.cnn(x)
        b,c,h,w = x.size()
        x = x.reshape(b,c*h,w).permute(0,2,1)
        x,_ = self.rnn(x)
        x = self.classifier(x)
        x = nn.functional.log_softmax(x, dim=2)
        return x.permute(1,0,2)

# =========================================================
# OCR Recognizer
# =========================================================

class CRNNRecognizer:
    def __init__(self, path, alphabet):
        self.device = Config.DEVICE
        self.int_to_char = {i+1:c for i,c in enumerate(alphabet)}
        self.int_to_char[0] = ''
        self.model = CRNN(len(alphabet)+1).to(self.device)
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        self.model.eval()
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Grayscale(),
            transforms.Resize((32,128)),
            transforms.ToTensor(),
            transforms.Normalize([0.5],[0.5])
        ])

    @torch.no_grad()
    def recognize(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        x = self.transform(img).unsqueeze(0).to(self.device)
        preds = self.model(x).permute(1,0,2)
        _, idx = torch.max(preds, 2)
        idx = idx.view(-1).cpu().numpy()
        res = []
        prev = 0
        for i in idx:
            if i != 0 and i != prev:
                res.append(self.int_to_char.get(int(i), ''))
            prev = i
        return ''.join(res)

# =========================================================
# MAIN GUI
# =========================================================

class ANPR_GUI:
    def __init__(self, root):
        self.root = root
        width, height = 910, 605
        screen_w = root.winfo_screenwidth()
        screen_h = root.winfo_screenheight()
        x = (screen_w//2) - (width//2)
        y = (screen_h//2) - (height//2)
        self.root.geometry(f"{width}x{height}+{x}+{y}")
        self.root.overrideredirect(True)
        self.root.configure(bg="#1b152c")

        self.current_user = None
        self.current_role = "user"
        self.sidebar = None

        print("Загрузка моделей...")
        self.detector = YOLO(Config.YOLO_MODEL_PATH)
        self.ocr = CRNNRecognizer(Config.OCR_MODEL_PATH, Config.OCR_ALPHABET)
        print("Модели загружены!")

        # Загрузка изображений интерфейса (должны лежать в папке)
        self.bg        = ImageTk.PhotoImage(Image.open("bg_main.png"))
        self.top_bar   = ImageTk.PhotoImage(Image.open("top_bar.png"))
        self.btn_upload   = ImageTk.PhotoImage(Image.open("btn_upload.png"))
        self.btn_upload_h = ImageTk.PhotoImage(Image.open("btn_upload_hover.png"))
        self.btn_manual   = ImageTk.PhotoImage(Image.open("btn_manual.png"))
        self.btn_manual_h = ImageTk.PhotoImage(Image.open("btn_manual_hover.png"))
        # Кнопки закрыть/свернуть
        self.btn_close   = ImageTk.PhotoImage(Image.open("btn_close.png"))
        self.btn_close_h = ImageTk.PhotoImage(Image.open("btn_close_hover.png"))
        self.btn_min     = ImageTk.PhotoImage(Image.open("btn_min.png"))
        self.btn_min_h   = ImageTk.PhotoImage(Image.open("btn_min_hover.png"))
        # Иконка в трее
        try:
            tray_img = PILImage.open("icon.ico")
        except:
            tray_img = PILImage.new("RGB", (64,64), "purple")
        self.tray_icon = pystray.Icon("AutoMooden", tray_img, "AutoMooden", menu=pystray.Menu(
            pystray.MenuItem("Открыть", self.restore_window),
            pystray.MenuItem("Выход", self.exit_app)
        ))

        self.canvas = tk.Canvas(root, width=910, height=605, highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)

        self.manual_number = tk.StringVar()
        self.result_image_id = None
        self.clear_btn_id = None

        self.show_login()

    # ======================== TRAY =========================
    def minimize_to_tray(self, event=None):
        self.root.withdraw()
        threading.Thread(target=self.tray_icon.run, daemon=True).start()

    def restore_window(self):
        self.root.after(0, self.root.deiconify)
        self.tray_icon.stop()

    def exit_app(self):
        self.tray_icon.stop()
        self.root.destroy()

    # ======================== LOGIN ========================
    def show_login(self):
        # Удаляем старые элементы с других экранов
        for attr in ['back_label', 'menu_btn']:
            if hasattr(self, attr):
                widget = getattr(self, attr)
                if widget and widget.winfo_exists():
                    widget.destroy()
        if self.sidebar:
            self.sidebar.destroy()
            self.sidebar = None

        self.canvas.delete("all")
        self.canvas.create_rectangle(0,0,910,605, fill="#241c3b", outline="")
        self.canvas.create_text(455,120, text="AutoMooDen", fill="#d7c8ff", font=("Arial",34,"bold"))
        self.canvas.create_text(455,180, text="Авторизация", fill="white", font=("Arial",20))

        login_var = tk.StringVar()
        pass_var = tk.StringVar()

        login_entry = tk.Entry(self.root, textvariable=login_var, font=("Arial",15),
                               justify="center", bg="#3b2d63", fg="white",
                               insertbackground="white", relief="flat")
        pass_entry = tk.Entry(self.root, textvariable=pass_var, show="*", font=("Arial",15),
                              justify="center", bg="#3b2d63", fg="white",
                              insertbackground="white", relief="flat")

        self.canvas.create_window(455,260, window=login_entry, width=260, height=42)
        self.canvas.create_window(455,320, window=pass_entry, width=260, height=42)

        def do_login():
            login = login_var.get().strip()
            password = pass_var.get().strip()
            if login in Config.USERS and Config.USERS[login]["password"] == password:
                self.current_user = login
                self.current_role = Config.USERS[login]["role"]
                self.show_main()
            else:
                messagebox.showerror("Ошибка", "Неверный логин или пароль")

        btn = tk.Button(self.root, text="Войти", bg="#7c63d6", fg="white",
                       font=("Arial",13,"bold"), relief="flat", cursor="hand2", command=do_login)
        self.canvas.create_window(455,410, window=btn, width=180, height=42)

    # ======================== MAIN =========================
    def show_main(self):
        # Очистка старых виджетов
        for attr in ['back_label', 'menu_btn']:
            if hasattr(self, attr):
                widget = getattr(self, attr)
                if widget and widget.winfo_exists():
                    widget.destroy()
        if self.sidebar:
            self.sidebar.destroy()
            self.sidebar = None

        self.canvas.delete("all")
        self.canvas.create_image(0,0, anchor=tk.NW, image=self.bg)
        self.top_bar_id = self.canvas.create_image(0,0, anchor=tk.NW, image=self.top_bar)

        # Кнопки закрыть/свернуть
        close_id = self.canvas.create_image(895,12, image=self.btn_close)
        min_id   = self.canvas.create_image(850,12, image=self.btn_min)
        self.add_hover(close_id, self.btn_close, self.btn_close_h)
        self.add_hover(min_id, self.btn_min, self.btn_min_h)
        self.canvas.tag_bind(close_id, "<Button-1>", lambda e: self.exit_app())
        self.canvas.tag_bind(min_id, "<Button-1>", self.minimize_to_tray)

        # Перетаскивание окна
        self.canvas.tag_bind(self.top_bar_id, "<ButtonPress-1>", self.start_move)
        self.canvas.tag_bind(self.top_bar_id, "<B1-Motion>", self.on_move)

        # Меню
        self.menu_btn = tk.Label(self.root, text="☰ Меню", bg="#6d56b3", fg="white",
                                 font=("Arial",11,"bold"), cursor="hand2", relief="flat",
                                 padx=10, pady=6)
        self.menu_btn.place(x=20, y=55)
        self.menu_btn.bind("<Button-1>", lambda e: self.toggle_sidebar())

        # Кнопка загрузки изображения
        self.upload_id = self.canvas.create_image(455,250, image=self.btn_upload)
        self.add_hover(self.upload_id, self.btn_upload, self.btn_upload_h)
        self.canvas.tag_bind(self.upload_id, "<Button-1>", self.load_image)

        # Ручной ввод номера
        self.manual_id = self.canvas.create_image(455,545, image=self.btn_manual)
        self.add_hover(self.manual_id, self.btn_manual, self.btn_manual_h)
        self.canvas.tag_bind(self.manual_id, "<Button-1>", self.toggle_manual_entry)

        self.manual_entry = tk.Entry(self.root, font=("Arial",14), justify="center",
                                     textvariable=self.manual_number, bg="#3b2d63",
                                     fg="white", insertbackground="white", relief="flat")
        self.confirm_btn = tk.Button(self.root, text="Открыть", bg="#6d56b3", fg="white",
                                     font=("Arial",11,"bold"), relief="flat",
                                     command=self.submit_number)
        self.manual_entry_window = self.canvas.create_window(455,560, window=self.manual_entry,
                                                             state="hidden", width=220)
        self.confirm_btn_window = self.canvas.create_window(455,592, window=self.confirm_btn,
                                                            state="hidden")

    # ======================== SIDEBAR ======================
    def toggle_sidebar(self):
        if self.sidebar:
            self.sidebar.destroy()
            self.sidebar = None
            return
        self.sidebar = tk.Frame(self.root, bg="#241c3b", width=220, height=560)
        self.sidebar.place(x=0, y=90)
        self.add_sidebar_btn("👤 Мой профиль", lambda: self.show_profile("A123BC"))
        if self.current_role == "moderator":
            self.add_sidebar_btn("🛡 Модерация", self.open_moderation)
        self.add_sidebar_btn("🚪 Выйти", self.logout)

    def add_sidebar_btn(self, text, cmd):
        btn = tk.Label(self.sidebar, text=text, bg="#34275a", fg="white",
                       font=("Arial",12,"bold"), padx=15, pady=12, anchor="w", cursor="hand2")
        btn.pack(fill="x", padx=10, pady=5)
        btn.bind("<Enter>", lambda e: btn.config(bg="#4d3b82"))
        btn.bind("<Leave>", lambda e: btn.config(bg="#34275a"))
        btn.bind("<Button-1>", lambda e: cmd())

    # ======================== OCR / IMAGE ==================
    def load_image(self, event=None):
        path = filedialog.askopenfilename()
        if not path:
            return
        img = cv2.imread(path)
        if img is None:
            return
        self.detect(img)

    def detect(self, img):
        res = self.detector(img, conf=0.3)
        detected_text = ""
        for r in res:
            for b in r.boxes:
                x1,y1,x2,y2 = map(int, b.xyxy[0])
                crop = img[y1:y2, x1:x2]
                text = self.ocr.recognize(crop)
                detected_text = text
                cv2.rectangle(img, (x1,y1), (x2,y2), (0,255,0), 2)
                cv2.putText(img, text, (x1,y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
        self.show_result(img)
        if detected_text:
            self.manual_number.set(detected_text)
        self.canvas.itemconfig(self.manual_entry_window, state="normal")
        self.canvas.itemconfig(self.confirm_btn_window, state="normal")

    def show_result(self, img):
        self.canvas.delete(self.upload_id)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img).resize((self.btn_upload.width(), self.btn_upload.height()))
        self.result_img = ImageTk.PhotoImage(img)
        self.result_image_id = self.canvas.create_image(455,250, image=self.result_img)
        self.canvas.tag_bind(self.result_image_id, "<Button-1>", self.load_image)
        # крестик для сброса
        x = 455 + self.btn_upload.width()//2 - 15
        y = 250 - self.btn_upload.height()//2 + 15
        self.clear_btn_id = self.canvas.create_text(x, y, text="✕", fill="red", font=("Arial",18,"bold"))
        self.canvas.tag_bind(self.clear_btn_id, "<Button-1>", self.clear_image)

    def clear_image(self, event=None):
        if self.result_image_id:
            self.canvas.delete(self.result_image_id)
            self.result_image_id = None
        if self.clear_btn_id:
            self.canvas.delete(self.clear_btn_id)
            self.clear_btn_id = None
        self.upload_id = self.canvas.create_image(455,250, image=self.btn_upload)
        self.add_hover(self.upload_id, self.btn_upload, self.btn_upload_h)
        self.canvas.tag_bind(self.upload_id, "<Button-1>", self.load_image)

    # ======================== PROFILE ======================
    def show_profile(self, number):
        self.canvas.delete("all")
        # удаляем оставшуюся кнопку Назад с предыдущего экрана
        if hasattr(self, 'back_label') and self.back_label and self.back_label.winfo_exists():
            self.back_label.destroy()
        user = self.get_user_data(number)   # если номера нет, вернёт пустую структуру

        # Шапка
        self.canvas.create_rectangle(0,0,910,170, fill="#5f4a92", outline="")
        # Кнопка Назад
        self.back_label = tk.Label(self.root, text="← Назад", bg="#6d56b3", fg="white",
                                   font=("Arial",12,"bold"), cursor="hand2", relief="flat",
                                   padx=10, pady=4)
        self.back_label.place(x=20, y=60)
        self.back_label.bind("<Button-1>", lambda e: self.show_main())

        self.canvas.create_text(455,70, text=number, fill="white", font=("Arial",34,"bold"))
        self.canvas.create_text(455,120, text=user["name"], fill="#e4dcff", font=("Arial",16))

        # Рейтинг
        self.canvas.create_rectangle(80,210,830,320, fill="#2b2147", outline="#cbbaff", width=3)
        self.canvas.create_text(180,250, text=f"{user['rating']:.1f}", fill="#ffd54f", font=("Arial",44,"bold"))
        stars = "★" * round(user["rating"])
        self.canvas.create_text(360,250, text=stars, fill="#ffd54f", font=("Arial",24))

        # Панель модератора
        if self.current_role == "moderator":
            rating_var = tk.DoubleVar(value=user["rating"])
            spin = tk.Spinbox(self.root, from_=0.0, to=5.0, increment=0.1,
                              textvariable=rating_var, width=5, font=("Arial",12))
            self.canvas.create_window(650,245, window=spin)
            save_btn = tk.Button(self.root, text="Сохранить", bg="#4d8f4d", fg="white",
                                 relief="flat",
                                 command=lambda: self.set_precise_rating(number, rating_var.get()))
            self.canvas.create_window(760,245, window=save_btn)

        # Предупреждение
        if user["warning"]:
            self.canvas.create_rectangle(80,340,830,390, fill="#471111", outline="#ff5b5b", width=3)
            self.canvas.create_text(455,365, text=f"⚠ {user['warning']}", fill="#ff9b9b", font=("Arial",16,"bold"))

        # Отзывы
        y = 420
        for idx, review in enumerate(user["reviews"]):
            if not review["approved"]:
                continue
            frame = tk.Frame(self.root, bg="#34275a", highlightbackground="#cbbaff", highlightthickness=2)
            tk.Label(frame, text=f"{review['author']} (оценка: {review['rating']})",
                     bg="#34275a", fg="white", font=("Arial",10,"bold")).pack(anchor="w", padx=10, pady=2)
            tk.Label(frame, text=review["text"], bg="#34275a", fg="#ddd",
                     wraplength=600, justify="left").pack(anchor="w", padx=10, pady=2)
            # теги в отзыве
            tags_str = " ".join([RATING_TAGS[t]["icon"] for t in review.get("tags", []) if t in RATING_TAGS])
            if tags_str:
                tk.Label(frame, text=tags_str, bg="#34275a", fg="#ffd54f", font=("Arial",10)).pack(anchor="w", padx=10, pady=2)

            if self.current_role == "moderator":
                del_btn = tk.Button(frame, text="Удалить", bg="#8b0000", fg="white", relief="flat",
                                    command=lambda i=idx: self.delete_review(number, i))
                del_btn.pack(anchor="e", padx=5, pady=5)
            self.canvas.create_window(455, y, window=frame, width=680, height=100)
            y += 110

        # Кнопка оставить отзыв
        review_btn = tk.Button(self.root, text="✏ Оставить отзыв", bg="#6d56b3", fg="white",
                               font=("Arial",12,"bold"), relief="flat",
                               command=lambda: self.open_review_form(number))
        self.canvas.create_window(455, y+30, window=review_btn, width=220, height=40)

    # ======================== REVIEW FORM ==================
    def open_review_form(self, number):
        win = tk.Toplevel(self.root)
        win.geometry("550x600")
        win.configure(bg="#241c3b")
        win.title("Новый отзыв")

        tk.Label(win, text="Ваш отзыв", bg="#241c3b", fg="white", font=("Arial",18,"bold")).pack(pady=10)

        text = tk.Text(win, height=6, bg="#34275a", fg="white", insertbackground="white", relief="flat")
        text.pack(fill="x", padx=20, pady=10)

        tk.Label(win, text="Оценка:", bg="#241c3b", fg="white", font=("Arial",12)).pack()
        rating_var = tk.DoubleVar(value=5.0)
        spin = tk.Spinbox(win, from_=0, to=5, increment=0.1, textvariable=rating_var,
                          font=("Arial",12), width=5)
        spin.pack(pady=5)

        tk.Label(win, text="Качества водителя:", bg="#241c3b", fg="white", font=("Arial",12)).pack(pady=(10,0))
        tags_vars = {}
        tags_frame = tk.Frame(win, bg="#241c3b")
        tags_frame.pack(pady=5)
        for tag_key, tag_info in RATING_TAGS.items():
            var = tk.BooleanVar()
            chk = tk.Checkbutton(tags_frame, text=f"{tag_info['icon']} {tag_info['text']}",
                                 variable=var, bg="#241c3b", fg="white", selectcolor="#6d56b3",
                                 font=("Arial",11))
            chk.pack(anchor="w")
            tags_vars[tag_key] = var

        def submit_review():
            txt = text.get("1.0", "end-1c").strip()
            if not txt:
                messagebox.showwarning("Ошибка", "Введите текст отзыва")
                return
            selected_tags = [tag for tag, var in tags_vars.items() if var.get()]
            # если номера нет в базе – создаём запись
            if number not in Config.TEMP_DB:
                Config.TEMP_DB[number] = {
                    "name": "",
                    "rating": 0.0,
                    "votes": 0,
                    "warning": "",
                    "reviews": []
                }
            # добавляем отзыв (пока не одобрен, если пользователь не модератор или всегда одобряем? По логике: обычный пользователь должен ждать одобрения)
            approved = (self.current_role == "moderator")
            Config.TEMP_DB[number]["reviews"].append({
                "author": self.current_user,
                "text": txt,
                "approved": approved,
                "rating": float(rating_var.get()),
                "tags": selected_tags
            })
            # пересчёт рейтинга (среднее по одобренным)
            self.update_rating(number)
            # обновляем счётчики тегов
            self.update_tag_counts(number)
            messagebox.showinfo("Готово", "Отзыв отправлен")
            win.destroy()
            self.show_profile(number)

        tk.Button(win, text="Отправить", bg="#6d56b3", fg="white", font=("Arial",12,"bold"),
                  relief="flat", command=submit_review).pack(pady=15)

    # ======================== MODERATION ===================
    def open_moderation(self):
        if self.current_role != "moderator":
            messagebox.showerror("Ошибка", "Нет доступа")
            return
        win = tk.Toplevel(self.root)
        win.geometry("750x550")
        win.configure(bg="#241c3b")
        win.title("Модерация отзывов")

        canvas = tk.Canvas(win, bg="#241c3b", highlightthickness=0)
        scrollbar = tk.Scrollbar(win, orient="vertical", command=canvas.yview)
        scroll_frame = tk.Frame(canvas, bg="#241c3b")
        scroll_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0,0), window=scroll_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        for num, data in Config.TEMP_DB.items():
            for idx, rev in enumerate(data["reviews"]):
                if rev["approved"]:
                    continue
                frame = tk.Frame(scroll_frame, bg="#34275a", highlightbackground="#cbbaff", highlightthickness=1)
                frame.pack(fill="x", padx=10, pady=5)
                tk.Label(frame, text=f"{num} | {rev['author']} | Оценка: {rev['rating']}",
                         bg="#34275a", fg="white", font=("Arial",11,"bold")).pack(anchor="w", padx=10, pady=2)
                tk.Label(frame, text=rev["text"], bg="#34275a", fg="#ddd", wraplength=600,
                         justify="left").pack(anchor="w", padx=10)
                # теги
                tags_str = " ".join([RATING_TAGS[t]["icon"] for t in rev.get("tags", []) if t in RATING_TAGS])
                if tags_str:
                    tk.Label(frame, text=tags_str, bg="#34275a", fg="#ffd54f", font=("Arial",10)).pack(anchor="w", padx=10)
                btn_frame = tk.Frame(frame, bg="#34275a")
                btn_frame.pack(anchor="e", pady=5, padx=5)
                tk.Button(btn_frame, text="Одобрить", bg="#4d8f4d", fg="white", relief="flat",
                          command=lambda n=num, i=idx: self.approve_and_refresh(n, i, win)).pack(side="left", padx=5)
                tk.Button(btn_frame, text="Удалить", bg="#8b0000", fg="white", relief="flat",
                          command=lambda n=num, i=idx: self.delete_and_refresh(n, i, win)).pack(side="left", padx=5)
        if not any(not rev["approved"] for data in Config.TEMP_DB.values() for rev in data["reviews"]):
            tk.Label(scroll_frame, text="Нет неодобренных отзывов", bg="#241c3b", fg="#aaa",
                     font=("Arial",14)).pack(pady=40)

    def approve_and_refresh(self, number, idx, win):
        Config.TEMP_DB[number]["reviews"][idx]["approved"] = True
        self.update_rating(number)
        self.update_tag_counts(number)
        win.destroy()
        self.open_moderation()  # переоткрыть окно модерации

    def delete_and_refresh(self, number, idx, win):
        del Config.TEMP_DB[number]["reviews"][idx]
        self.update_rating(number)
        self.update_tag_counts(number)
        win.destroy()
        self.open_moderation()

    def delete_review(self, number, idx):
        """Удаление прямо из профиля (модератором)"""
        del Config.TEMP_DB[number]["reviews"][idx]
        self.update_rating(number)
        self.update_tag_counts(number)
        self.show_profile(number)   # обновить страницу

    # ======================== LOGIC =======================
    def update_rating(self, number):
        """Пересчитывает рейтинг как среднее по одобренным отзывам"""
        if number not in Config.TEMP_DB:
            return
        approved = [r for r in Config.TEMP_DB[number]["reviews"] if r["approved"]]
        if approved:
            avg = sum(r["rating"] for r in approved) / len(approved)
            Config.TEMP_DB[number]["rating"] = round(avg, 1)
            Config.TEMP_DB[number]["votes"] = len(approved)
        else:
            Config.TEMP_DB[number]["rating"] = 0.0
            Config.TEMP_DB[number]["votes"] = 0

    def set_precise_rating(self, number, value):
        """Модератор вручную задаёт рейтинг"""
        if number in Config.TEMP_DB:
            Config.TEMP_DB[number]["rating"] = round(float(value), 1)
            messagebox.showinfo("Готово", "Рейтинг изменён")
            self.show_profile(number)

    def update_tag_counts(self, number):
        """Подсчитывает, сколько раз каждый тег встречается в одобренных отзывах,
           и сохраняет в data['tag_counts'] для отображения иконок рядом с номером."""
        if number not in Config.TEMP_DB:
            return
        counts = {tag:0 for tag in RATING_TAGS}
        for rev in Config.TEMP_DB[number]["reviews"]:
            if rev["approved"]:
                for t in rev.get("tags", []):
                    if t in counts:
                        counts[t] += 1
        Config.TEMP_DB[number]["tag_counts"] = counts

    def get_user_data(self, number):
        """Возвращает копию данных, гарантируя наличие ключа 'tag_counts'"""
        if number not in Config.TEMP_DB:
            # создаём запись по умолчанию, если её нет
            Config.TEMP_DB[number] = {
                "name": "",
                "rating": 0.0,
                "votes": 0,
                "warning": "",
                "reviews": [],
                "tag_counts": {}
            }
        data = copy.deepcopy(Config.TEMP_DB[number])
        if "tag_counts" not in data:
            data["tag_counts"] = {}
        return data

    def get_tag_icons_for_number(self, number):
        """Возвращает строку из иконок тегов, которые превысили порог"""
        data = self.get_user_data(number)
        counts = data.get("tag_counts", {})
        icons = []
        for tag, info in RATING_TAGS.items():
            if counts.get(tag, 0) >= info["threshold"]:
                icons.append(info["icon"])
        return " ".join(icons)

    # ======================== HELPERS ======================
    def logout(self):
        self.current_user = None
        self.current_role = "user"
        if self.sidebar:
            self.sidebar.destroy()
            self.sidebar = None
        self.show_login()

    def submit_number(self):
        number = self.manual_number.get().strip()
        if number:
            self.show_profile(number)

    def toggle_manual_entry(self, event=None):
        cur = self.canvas.itemcget(self.manual_entry_window, "state")
        if cur == "hidden":
            self.canvas.itemconfig(self.manual_entry_window, state="normal")
            self.canvas.itemconfig(self.confirm_btn_window, state="normal")
        else:
            self.canvas.itemconfig(self.manual_entry_window, state="hidden")
            self.canvas.itemconfig(self.confirm_btn_window, state="hidden")

    def add_hover(self, item, normal, hover):
        def enter(e): self.canvas.itemconfig(item, image=hover)
        def leave(e): self.canvas.itemconfig(item, image=normal)
        self.canvas.tag_bind(item, "<Enter>", enter)
        self.canvas.tag_bind(item, "<Leave>", leave)

    def start_move(self, event):
        self.offset_x = event.x
        self.offset_y = event.y

    def on_move(self, event):
        x = self.root.winfo_pointerx() - self.offset_x
        y = self.root.winfo_pointery() - self.offset_y
        self.root.geometry(f"+{x}+{y}")

# =========================================================
# START
# =========================================================
if __name__ == "__main__":
    root = tk.Tk()
    app = ANPR_GUI(root)
    root.mainloop()