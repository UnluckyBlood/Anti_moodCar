import tkinter as tk
from tkinter import filedialog
import easyocr
import cv2
import re

class NumberPlateReader:
    def __init__(self, root):
        self.root = root
        self.root.title("Распознавание номера")
        self.root.geometry("520x260")

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

    # 🔧 исправление символов
    def fix_chars(self, text):
        replace = {
            '0': 'O',
            '1': 'I',
            '8': 'B',
            'P': 'P',  # оставляем, но дальше проверим
            'О': 'O',
            'С': 'C',
            'А': 'A',
            'В': 'B',
            'Е': 'E',
            'К': 'K',
            'М': 'M',
            'Н': 'H',
            'Р': 'P',
            'Т': 'T',
            'У': 'Y',
            'Х': 'X'
        }
        for k, v in replace.items():
            text = text.replace(k, v)
        return text

    # 🔥 сборка номера
    def parse_plate(self, words):
        letters = "ABEKMHOPCTYX"

        for i in range(len(words)):
            for j in range(i+1, min(i+4, len(words))):
                chunk = "".join(words[i:j+1])

                chunk = self.fix_chars(chunk)
                chunk = re.sub(r'[^A-Z0-9]', '', chunk)

                if len(chunk) >= 6:
                    # пробуем формат: L DDD LL
                    if (chunk[0] in letters and
                        chunk[1:4].isdigit() and
                        all(c in letters for c in chunk[4:6])):
                        return chunk

        return None

    def run(self):
        img = cv2.imread(self.image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # усиливаем контраст
        gray = cv2.resize(gray, None, fx=2, fy=2)
        gray = cv2.equalizeHist(gray)

        result = self.reader.readtext(gray, detail=0)

        words = []
        for r in result:
            words += r.split()

        plate = self.parse_plate(words)

        self.out.delete(1.0, tk.END)

        if plate:
            self.out.insert(tk.END, f"Номер: {plate}")
        else:
            self.out.insert(tk.END, "Не удалось точно определить\n\n" + " ".join(words))


if __name__ == "__main__":
    root = tk.Tk()
    app = NumberPlateReader(root)
    root.mainloop()