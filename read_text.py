import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image
import easyocr

class NumberPlateReader:
    def __init__(self, root):
        self.root = root
        self.root.title("Чтение номера машины")
        self.root.geometry("500x200")

        self.reader = easyocr.Reader(['ru', 'en'])  # русский и английский
        self.image_path = None

        # Кнопка загрузки
        self.btn_load = tk.Button(root, text="Загрузить картинку", command=self.load_image)
        self.btn_load.pack(pady=10)

        # Кнопка распознавания
        self.btn_recognize = tk.Button(root, text="Распознать текст", command=self.recognize_text, state=tk.DISABLED)
        self.btn_recognize.pack(pady=5)

        # Метка с путём к файлу
        self.lbl_path = tk.Label(root, text="Файл не выбран", fg="gray")
        self.lbl_path.pack()

        # Поле для вывода результата
        self.result_text = tk.Text(root, height=4, width=50)
        self.result_text.pack(pady=10)

    def load_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Изображения", "*.jpg *.jpeg *.png *.bmp *.tiff")]
        )
        if file_path:
            self.image_path = file_path
            self.lbl_path.config(text=file_path, fg="black")
            self.btn_recognize.config(state=tk.NORMAL)
        else:
            self.lbl_path.config(text="Файл не выбран", fg="gray")
            self.btn_recognize.config(state=tk.DISABLED)

    def recognize_text(self):
        if not self.image_path:
            return
        try:
            # Простое распознавание всего текста на изображении
            result = self.reader.readtext(self.image_path, detail=0)  # только текст
            text = " ".join(result) if result else "Текст не найден"
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(tk.END, text)
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось распознать: {e}")

if __name__ == "__main__":
    root = tk.Tk()
    app = NumberPlateReader(root)
    root.mainloop()