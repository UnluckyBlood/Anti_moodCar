import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image
import easyocr
import cv2
import numpy as np
import os
import urllib.request
import re

CASCADE_URL = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_russian_plate_number.xml"
CASCADE_FILE = "haarcascade_russian_plate_number.xml"

class NumberPlateReader:
    def __init__(self, root):
        self.root = root
        self.root.title("Чтение номера авто")
        self.root.geometry("550x280")

        self.reader = easyocr.Reader(['ru', 'en'])
        self.image_path = None

        self.btn_load = tk.Button(root, text="Загрузить картинку", command=self.load_image)
        self.btn_load.pack(pady=10)

        self.btn_recognize = tk.Button(root, text="Распознать номер", command=self.recognize_plate, state=tk.DISABLED)
        self.btn_recognize.pack(pady=5)

        self.lbl_path = tk.Label(root, text="Файл не выбран", fg="gray")
        self.lbl_path.pack()

        self.result_text = tk.Text(root, height=3, width=50)
        self.result_text.pack(pady=10)

        self.lbl_status = tk.Label(root, text="", fg="blue")
        self.lbl_status.pack()

        self._ensure_cascade()

    def _ensure_cascade(self):
        if not os.path.exists(CASCADE_FILE):
            try:
                urllib.request.urlretrieve(CASCADE_URL, CASCADE_FILE)
                print("Каскад скачан.")
            except Exception as e:
                messagebox.showerror("Ошибка", f"Не удалось скачать каскад Хаара:\n{e}")
                return
        self.plate_cascade = cv2.CascadeClassifier(CASCADE_FILE)

    def load_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Изображения", "*.jpg *.jpeg *.png *.bmp *.tiff")]
        )
        if file_path:
            self.image_path = file_path
            self.lbl_path.config(text=file_path, fg="black")
            self.btn_recognize.config(state=tk.NORMAL)
            self.lbl_status.config(text="")
        else:
            self.lbl_path.config(text="Файл не выбран", fg="gray")
            self.btn_recognize.config(state=tk.DISABLED)

    def recognize_plate(self):
        if not self.image_path:
            return
        self.result_text.delete(1.0, tk.END)
        self.lbl_status.config(text="Ищу номерную пластину...")
        self.root.update()

        img = cv2.imread(self.image_path)
        if img is None:
            messagebox.showerror("Ошибка", "Не могу открыть файл")
            return

        # Предобработка для улучшения детекции
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)

        # Расширенный поиск пластин (более чувствительные параметры)
        plates = self.plate_cascade.detectMultiScale(
            enhanced,
            scaleFactor=1.05,          # более плавное изменение масштаба
            minNeighbors=3,            # снижено для повышения чувствительности
            minSize=(20, 10)           # минимальный размер пластины
        )

        print(f"Найдено областей, похожих на номер: {len(plates)}")  # отладка в консоль

        if len(plates) > 0:
            # Выбираем самую крупную пластину (ближайшую)
            max_area = 0
            best_plate = None
            for (x, y, w, h) in plates:
                area = w * h
                if area > max_area:
                    max_area = area
                    best_plate = (x, y, w, h)
            if best_plate:
                x, y, w, h = best_plate
                plate_img = enhanced[y:y+h, x:x+w]  # используем улучшенный фрагмент
                plate_rgb = cv2.cvtColor(plate_img, cv2.COLOR_GRAY2RGB)
                plate_pil = Image.fromarray(plate_rgb)

                result = self.reader.readtext(np.array(plate_pil), detail=0)
                text = " ".join(result).strip() if result else "Не удалось прочитать"
                self.result_text.insert(tk.END, text)
                self.lbl_status.config(text=f"Пластина найдена, результат: {text}", fg="green")
                return

        # Если каскад не нашёл пластину, пробуем найти текст на всём изображении
        # и отфильтровать по шаблону российского номера (А123ВС) как запасной вариант
        full_result = self.reader.readtext(enhanced, detail=0)
        full_text = " ".join(full_result) if full_result else ""

        # Простой фильтр: ищем комбинацию букв и цифр, похожую на номер
        # Шаблон: одна буква, три цифры, две буквы (может быть с пробелами)
        plate_candidates = re.findall(r'[АВЕКМНОРСТУХA-Z]\s?\d{3}\s?[АВЕКМНОРСТУХA-Z]{2}', full_text, re.IGNORECASE)
        if plate_candidates:
            # Убираем пробелы и объединяем
            cleaned = [re.sub(r'\s+', '', c) for c in plate_candidates]
            text = ", ".join(cleaned)
            self.result_text.insert(tk.END, text)
            self.lbl_status.config(text="Номер определён по тексту со всей картинки (без пластины)", fg="orange")
        else:
            if full_text:
                self.result_text.insert(tk.END, full_text)
                self.lbl_status.config(text="Пластина не найдена; весь текст на фото:", fg="orange")
            else:
                self.result_text.insert(tk.END, "Ничего не распознано")
                self.lbl_status.config(text="Пластина не найдена, текст отсутствует", fg="red")

if __name__ == "__main__":
    try:
        root = tk.Tk()
        app = NumberPlateReader(root)
        root.mainloop()
    except Exception as e:
        import traceback
        print("Ошибка при создании окна:")
        traceback.print_exc()
        input("Нажмите Enter для выхода...")