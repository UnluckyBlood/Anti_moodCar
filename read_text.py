import tkinter as tk
from tkinter import filedialog
import easyocr
import cv2
import re
import numpy as np

class NumberPlateReader:
    def __init__(self, root):
        self.root = root
        self.root.title("Распознавание номера")
        self.root.geometry("520x260")

        # Разрешённые символы российских номеров
        self.allowlist = 'ABEKMHOPCTYX0123456789'
        self.reader = easyocr.Reader(['ru', 'en'])
        self.image_path = None

        tk.Button(root, text="Загрузить", command=self.load).pack(pady=10)
        self.btn = tk.Button(root, text="Распознать", command=self.run, state=tk.DISABLED)
        self.btn.pack()

        self.out = tk.Text(root, height=5, width=60)
        self.out.pack(pady=10)

    def load(self):
        f = filedialog.askopenfilename(filetypes=[("img", "*.jpg *.png *.jpeg")])
        if f:
            self.image_path = f
            self.btn.config(state=tk.NORMAL)

    def preprocess(self, img):
        """Улучшаем изображение для OCR"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

        # Адаптивная бинаризация (белый текст на чёрном или наоборот)
        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )
        # Можно инвертировать, если фон тёмный – добавим обе версии
        inverted = cv2.bitwise_not(thresh)
        return gray, thresh, inverted

    def parse_plate(self, text):
        """Ищем номер в распознанной строке с учётом шаблона"""
        # Оставляем только разрешённые символы
        clean = re.sub(r'[^ABEKMHOPCTYX0-9]', '', text.upper())

        # Шаблон: буква + 3 цифры + 2 буквы + 2-3 цифры региона
        pattern = r'([ABEKMHOPCTYX])(\d{3})([ABEKMHOPCTYX]{2})(\d{2,3})'
        match = re.search(pattern, clean)
        if match:
            l1, d, l2, rgn = match.groups()
            # Исправляем типичные ошибки OCR в буквенных позициях
            fix_letter = {'0': 'O', '8': 'B', '5': 'S'}  # S не используется, но на всякий
            l1 = ''.join(fix_letter.get(ch, ch) for ch in l1)
            l2 = ''.join(fix_letter.get(ch, ch) for ch in l2)
            # В цифровых позициях наоборот – буквы, похожие на цифры, заменяем на цифры
            fix_digit = {'O': '0', 'B': '8', 'I': '1', 'Z': '2', 'S': '5'}
            d = ''.join(fix_digit.get(ch, ch) for ch in d)
            rgn = ''.join(fix_digit.get(ch, ch) for ch in rgn)
            return f"{l1}{d}{l2}{rgn}"
        return None

    def run(self):
        img = cv2.imread(self.image_path)
        if img is None:
            self.out.delete(1.0, tk.END)
            self.out.insert(tk.END, "Ошибка загрузки изображения")
            return

        gray, thresh, inverted = self.preprocess(img)

        # Пробуем распознать на трёх вариантах и объединяем результаты
        results = set()
        for image in [gray, thresh, inverted]:
            # allowlist задаём при первом вызове (можно передавать в readtext)
            res = self.reader.readtext(image, detail=0, allowlist=self.allowlist)
            # Объединяем все куски в одну строку
            full_text = ''.join(res).replace(' ', '')
            plate = self.parse_plate(full_text)
            if plate:
                results.add(plate)

        self.out.delete(1.0, tk.END)
        if results:
            # Если нашли несколько, берём самый длинный (с регионом)
            plate = max(results, key=len)
            self.out.insert(tk.END, f"Номер: {plate}")
        else:
            self.out.insert(tk.END, "Номер не найден")

if __name__ == "__main__":
    root = tk.Tk()
    app = NumberPlateReader(root)
    root.mainloop()