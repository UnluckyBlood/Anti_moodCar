import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import cv2
from PIL import Image, ImageTk
from ultralytics import YOLO
import torch
import torch.nn as nn
from torchvision import transforms
import pystray
from PIL import Image as PILImage
import copy

# ────────── Конфиг ──────────
class Config:
    YOLO_MODEL_PATH = "best.pt"
    OCR_MODEL_PATH = "crnn_ocr_model.pth"
    OCR_IMG_HEIGHT = 32
    OCR_IMG_WIDTH = 128
    OCR_ALPHABET = '0123456789ABCEHKMOPTXY'
    DEVICE = torch.device("cpu")

    # Временная БД (потом заменим на PostgreSQL)
    TEMP_DB = {
        "A123BC": {
            "name": "Иван Иванов",
            "rating": 4.5,
            "reviews": [
                {"author": "Петр", "text": "Отличный водитель!", "approved": True},
                {"author": "Мария", "text": "Паркуется как бог", "approved": False}
            ]
        },
        "B777OO": {
            "name": "Сергей Смирнов",
            "rating": 3.2,
            "reviews": [
                {"author": "Аноним", "text": "Нормально", "approved": True}
            ]
        }
    }

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

        # Загрузка изображений
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

        self.canvas=tk.Canvas(root,width=910,height=605,highlightthickness=0)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Поле ручного ввода (создаётся один раз)
        self.manual_number = tk.StringVar()
        self.manual_entry = tk.Entry(self.root, font=("Arial", 14), justify="center",
                                     textvariable=self.manual_number)
        # Кнопка подтверждения (создаётся один раз)
        self.confirm_btn = tk.Button(self.root, text="Подтвердить", font=("Arial", 12),
                                     command=self.submit_number)

        # Вспомогательные идентификаторы
        self.result_image_id = None
        self.clear_btn_id = None
        self.profile_page_active = False
        self.scrollbar = None
        self.back_btn = None

        # Трей
        try:
            tray_img = PILImage.open("icon.ico")
        except:
            tray_img = PILImage.new("RGB", (64,64), "blue")
        self.tray_menu = pystray.Menu(
            pystray.MenuItem("Открыть", self.show_window),
            pystray.MenuItem("Модерация", self.open_moderation),
            pystray.MenuItem("Выход", self.quit_app)
        )
        self.tray_icon = pystray.Icon("ANPR", tray_img, "ANPR", self.tray_menu)
        self.tray_thread = None

        self.root.protocol('WM_DELETE_WINDOW', self.quit_app)

        # Построить главный экран
        self.show_main()

    # ────────── Главный экран ──────────
    def show_main(self):
        # Сброс режима профиля
        if self.profile_page_active:
            self.canvas.unbind_all("<MouseWheel>")
            self.canvas.unbind_all("<Button-4>")
            self.canvas.unbind_all("<Button-5>")
            if self.scrollbar:
                self.scrollbar.place_forget()
                self.scrollbar.destroy()
                self.scrollbar = None
            if self.back_btn:
                self.back_btn.place_forget()
                self.back_btn.destroy()
                self.back_btn = None
            self.profile_page_active = False

        # Очистка canvas и возврат к фиксированному размеру
        self.canvas.delete("all")
        self.canvas.config(width=910, height=605, yscrollcommand="")

        # Фон и верхняя панель
        self.canvas.create_image(0,0,anchor=tk.NW,image=self.bg)
        self.top_bar_id = self.canvas.create_image(0,0,anchor=tk.NW,image=self.top_bar)

        # Кнопки
        self.upload_id = self.canvas.create_image(455,257,image=self.btn_upload)
        self.manual_id = self.canvas.create_image(455,545,image=self.btn_manual)
        self.close_id = self.canvas.create_image(899,12,image=self.btn_close)
        self.min_id = self.canvas.create_image(850,12,image=self.btn_min)

        # Поле ввода и кнопка подтверждения (скрыты)
        self.manual_entry_window = self.canvas.create_window(
            455, 560, window=self.manual_entry, state="hidden", width=200
        )
        self.confirm_btn_window = self.canvas.create_window(
            455, 590, window=self.confirm_btn, state="hidden"
        )

        # Кнопка модерации (текст)
        self.moderation_id = self.canvas.create_text(
            800, 30, text="Мод.", font=("Arial", 10, "underline"), fill="white",
            activefill="lightgray"
        )
        self.canvas.tag_bind(self.moderation_id, "<Button-1>", lambda e: self.open_moderation())

        # Ховер-эффекты
        self.add_hover(self.upload_id, self.btn_upload, self.btn_upload_h)
        self.add_hover(self.manual_id, self.btn_manual, self.btn_manual_h)
        self.add_hover(self.close_id, self.btn_close, self.btn_close_h)
        self.add_hover(self.min_id, self.btn_min, self.btn_min_h)

        # События
        self.canvas.tag_bind(self.upload_id, "<Button-1>", self.load_image)
        self.canvas.tag_bind(self.manual_id, "<Button-1>", self.toggle_manual_entry)
        self.canvas.tag_bind(self.close_id, "<Button-1>", lambda e: self.root.destroy())
        self.canvas.tag_bind(self.min_id, "<Button-1>", self.minimize_to_tray)
        self.canvas.tag_bind(self.top_bar_id, "<ButtonPress-1>", self.start_move)
        self.canvas.tag_bind(self.top_bar_id, "<B1-Motion>", self.on_move)

        self.offset_x = 0
        self.offset_y = 0
        self.result_image_id = None
        self.clear_btn_id = None

    # ────────── Страница профиля ──────────
    def show_profile(self, number):
        self.profile_page_active = True
        self.canvas.delete("all")

        # Настройка прокрутки
        self.canvas.config(width=895, height=605)
        self.scrollbar = tk.Scrollbar(self.root, orient=tk.VERTICAL, command=self.canvas.yview)
        self.scrollbar.place(x=895, y=0, height=605)
        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        # Кнопка «Назад»
        self.back_btn = tk.Label(self.root, text="← Назад", font=("Arial", 12, "underline"),
                                 fg="blue", bg="#f0f0f0", cursor="hand2")
        self.back_btn.place(x=10, y=10)
        self.back_btn.bind("<Button-1>", lambda e: self.show_main())

        # Прокручиваемая область
        profile_frame = tk.Frame(self.canvas, bg='#f0f0f0')
        self.canvas.create_window((0, 0), window=profile_frame, anchor='nw')
        profile_frame.bind("<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))

        # Данные пользователя
        user_data = self.get_user_data(number)

        # Номер
        tk.Label(profile_frame, text=number, font=("Arial", 28, "bold"),
                 bg='#f0f0f0').pack(pady=(30, 5))

        # Имя
        name = user_data.get("name", "")
        if name:
            tk.Label(profile_frame, text=name, font=("Arial", 16), bg='#f0f0f0').pack(pady=5)
        else:
            tk.Label(profile_frame, text="Имя не указано", font=("Arial", 16, "italic"),
                     fg="gray", bg='#f0f0f0').pack(pady=5)

        # Рейтинг
        rating = user_data.get("rating", 0.0)
        tk.Label(profile_frame, text=f"Рейтинг: {rating}", font=("Arial", 14),
                 bg='#f0f0f0').pack(pady=5)

        # Разделитель
        ttk.Separator(profile_frame, orient='horizontal').pack(fill='x', padx=20, pady=15)

        # Блок отзывов
        reviews_frame = tk.Frame(profile_frame, bg='#f0f0f0')
        reviews_frame.pack(fill='both', expand=True, padx=20, pady=10)

        def refresh_reviews():
            for widget in reviews_frame.winfo_children():
                widget.destroy()
            approved = [r for r in user_data["reviews"] if r["approved"]]
            if not approved:
                tk.Label(reviews_frame, text="Пока нет отзывов", bg='#f0f0f0',
                         font=("Arial", 10)).pack()
            for rev in approved:
                cont = tk.Frame(reviews_frame, bg='white', relief='groove', bd=2)
                cont.pack(fill='x', pady=5, padx=5)
                tk.Label(cont, text=f"Автор: {rev['author']}", bg='white',
                         font=("Arial", 10, "bold")).pack(anchor='w')
                tk.Label(cont, text=rev['text'], bg='white', font=("Arial", 9),
                         wraplength=780, justify='left').pack(anchor='w', padx=10, pady=5)

        refresh_reviews()

        # Кнопка добавления отзыва
        review_btn = tk.Button(profile_frame, text="Оставить отзыв",
                               command=lambda: toggle_review_form())
        review_btn.pack(pady=10)

        # Форма отзыва (скрыта)
        form_visible = False
        review_form_frame = tk.Frame(profile_frame, bg='#e0e0e0', relief='sunken', bd=2)

        def toggle_review_form():
            nonlocal form_visible
            if form_visible:
                review_form_frame.pack_forget()
                form_visible = False
            else:
                review_form_frame.pack(before=review_btn, fill='x', padx=20, pady=10)
                form_visible = True

        tk.Label(review_form_frame, text="Ваше имя:", bg='#e0e0e0').pack(anchor='w', padx=5, pady=2)
        author_entry = tk.Entry(review_form_frame, width=30)
        author_entry.pack(padx=5, pady=2)
        author_entry.insert(0, "Аноним")
        tk.Label(review_form_frame, text="Текст отзыва:", bg='#e0e0e0').pack(anchor='w', padx=5, pady=2)
        text_entry = tk.Text(review_form_frame, height=4, width=60)
        text_entry.pack(padx=5, pady=2)

        def submit_review():
            author = author_entry.get().strip()
            text = text_entry.get("1.0", "end-1c").strip()
            if not text:
                return
            self.add_review(number, author if author else "Аноним", text)
            review_form_frame.pack_forget()
            nonlocal form_visible
            form_visible = False
            # обновляем список отзывов
            nonlocal user_data
            user_data = self.get_user_data(number)
            refresh_reviews()
            messagebox.showinfo("Отзыв", "Отзыв отправлен на модерацию")

        tk.Button(review_form_frame, text="Отправить", command=submit_review).pack(pady=5)

        # Прокрутка колесом мыши
        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel_profile)
        self.canvas.bind_all("<Button-4>", self._on_mousewheel_profile)
        self.canvas.bind_all("<Button-5>", self._on_mousewheel_profile)

        self.root.update_idletasks()
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def _on_mousewheel_profile(self, event):
        if self.profile_page_active:
            if event.num == 4 or event.delta > 0:
                self.canvas.yview_scroll(-1, "units")
            elif event.num == 5 or event.delta < 0:
                self.canvas.yview_scroll(1, "units")

    # ────────── Подтверждение номера ──────────
    def submit_number(self):
        number = self.manual_number.get().strip()
        if number:
            self.show_profile(number)

    # ────────── Работа с данными ──────────
    def get_user_data(self, number):
        if number in Config.TEMP_DB:
            return copy.deepcopy(Config.TEMP_DB[number])
        return {"name": "", "rating": 0.0, "reviews": []}

    def add_review(self, number, author, text):
        if number not in Config.TEMP_DB:
            Config.TEMP_DB[number] = {"name": "", "rating": 0.0, "reviews": []}
        Config.TEMP_DB[number]["reviews"].append(
            {"author": author, "text": text, "approved": False}
        )

    def approve_review(self, number, idx, callback=None):
        if number in Config.TEMP_DB and idx < len(Config.TEMP_DB[number]["reviews"]):
            Config.TEMP_DB[number]["reviews"][idx]["approved"] = True
            if callback:
                callback()

    def delete_review(self, number, idx, callback=None):
        if number in Config.TEMP_DB and idx < len(Config.TEMP_DB[number]["reviews"]):
            del Config.TEMP_DB[number]["reviews"][idx]
            if callback:
                callback()

    # ────────── Модерация отзывов ──────────
    def open_moderation(self):
        win = tk.Toplevel(self.root)
        win.title("Модерация отзывов")
        win.geometry("600x400")

        main_frame = tk.Frame(win)
        main_frame.pack(fill='both', expand=True)

        canvas = tk.Canvas(main_frame, bg='white')
        scrollbar = tk.Scrollbar(main_frame, orient='vertical', command=canvas.yview)
        scrollable = tk.Frame(canvas, bg='white')

        scrollable.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0,0), window=scrollable, anchor='nw')
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side='left', fill='both', expand=True)
        scrollbar.pack(side='right', fill='y')

        def refresh():
            for w in scrollable.winfo_children():
                w.destroy()
            pending = []
            for num, data in Config.TEMP_DB.items():
                for idx, rev in enumerate(data["reviews"]):
                    if not rev["approved"]:
                        pending.append((num, idx, rev))
            if not pending:
                tk.Label(scrollable, text="Нет отзывов на модерации", bg='white').pack(pady=10)
            for num, idx, rev in pending:
                f = tk.Frame(scrollable, bg='#f9f9f9', relief='groove', bd=2)
                f.pack(fill='x', padx=5, pady=5)
                tk.Label(f, text=f"Номер: {num}  |  Автор: {rev['author']}",
                         bg='#f9f9f9', font=('Arial', 10, 'bold')).pack(anchor='w')
                tk.Label(f, text=rev['text'], bg='#f9f9f9', wraplength=500,
                         justify='left').pack(anchor='w', padx=10)
                btn_frame = tk.Frame(f, bg='#f9f9f9')
                btn_frame.pack(anchor='e', padx=5, pady=2)
                tk.Button(btn_frame, text="Одобрить",
                          command=lambda n=num, i=idx: (self.approve_review(n, i), refresh())
                         ).pack(side='left', padx=2)
                tk.Button(btn_frame, text="Удалить",
                          command=lambda n=num, i=idx: (self.delete_review(n, i), refresh())
                         ).pack(side='left', padx=2)
        refresh()

    # ────────── Вспомогательные методы главного экрана ──────────
    def add_hover(self, item, normal, hover):
        def on_enter(e): self.canvas.itemconfig(item,image=hover)
        def on_leave(e): self.canvas.itemconfig(item,image=normal)
        self.canvas.tag_bind(item,"<Enter>",on_enter)
        self.canvas.tag_bind(item,"<Leave>",on_leave)

    def start_move(self,event):
        self.offset_x=event.x
        self.offset_y=event.y

    def on_move(self,event):
        x=self.root.winfo_pointerx()-self.offset_x
        y=self.root.winfo_pointery()-self.offset_y
        self.root.geometry(f"+{x}+{y}")

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
                detected_text = text
                cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
                cv2.putText(img,text,(x1,y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0),2)
        self.show_result(img)
        if detected_text:
            self.manual_number.set(detected_text)
        # Показываем поле ввода и кнопку подтверждения
        self.canvas.itemconfig(self.manual_entry_window, state="normal")
        self.canvas.itemconfig(self.confirm_btn_window, state="normal")
        self.manual_entry.focus_set()

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

        self.canvas.itemconfig(self.manual_entry_window, state="hidden")
        self.canvas.itemconfig(self.confirm_btn_window, state="hidden")
        self.manual_number.set("")

    def toggle_manual_entry(self, event=None):
        cur_state = self.canvas.itemcget(self.manual_entry_window, "state")
        if cur_state == "hidden":
            self.canvas.itemconfig(self.manual_entry_window, state="normal")
            self.canvas.itemconfig(self.confirm_btn_window, state="normal")
            self.manual_entry.focus_set()
        else:
            self.canvas.itemconfig(self.manual_entry_window, state="hidden")
            self.canvas.itemconfig(self.confirm_btn_window, state="hidden")
            self.manual_number.set("")

    # ────────── Сворачивание в трей ──────────
    def minimize_to_tray(self, event=None):
        self.root.withdraw()
        if not self.tray_thread or not self.tray_thread.is_alive():
            import threading
            self.tray_thread = threading.Thread(target=self.tray_icon.run, daemon=True)
            self.tray_thread.start()

    def show_window(self):
        self.root.deiconify()
        if self.tray_icon:
            self.tray_icon.stop()
        self.tray_thread = None

    def quit_app(self):
        if self.tray_icon:
            self.tray_icon.stop()
        self.root.destroy()

if __name__=="__main__":
    root=tk.Tk()
    app=ANPR_GUI(root)
    root.mainloop()