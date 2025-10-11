import tkinter as tk
from pathlib import Path
from tkinter import filedialog
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

SUPPORTED_IMAGE_TYPES = [("Изображения", "*.png;*.jpg;*.jpeg;*.bmp;*.gif")]

def read_image():
    global image
    filepath = filedialog.askopenfilename(filetypes=SUPPORTED_IMAGE_TYPES)
    if not filepath:
        return None

    path= Path(filepath)
    try:
        image = Image.open(path).convert("RGB")
    except Exception as exception:
        status_label.config(text=f"Ошибка загрузки: {exception}")
        return None

    status_label.config(text=f"Файл загружен: {path.name}")
    return image


def count_rgb():
    if image is None:
        status_label.config(text="Сначала загрузите изображение!")
        return

    arr = np.asarray(image, dtype=np.uint8)

    r = arr[:, :, 0].ravel()
    g = arr[:, :, 1].ravel()
    b = arr[:, :, 2].ravel()

    plt.figure("Гистограмма RGB")
    plt.hist([r, g, b], bins=256, color=["red", "green", "blue"],
             alpha=0.5, label=["Red", "Green", "Blue"])
    plt.xlabel("Значение канала (0-255)")
    plt.ylabel("Количество пикселей")
    plt.legend()
    plt.tight_layout()
    plt.show()
    plt.close()


root = tk.Tk()
root.title("Подсчёт RGB пикселей")
root.geometry("400x200")

load_btn = tk.Button(root, text="Загрузить изображение", command=read_image)
load_btn.pack(pady=10)

count_btn = tk.Button(root, text="Показать график RGB", command=count_rgb)
count_btn.pack(pady=10)

status_label = tk.Label(root, text="Изображение не загружено", fg="blue")
status_label.pack(pady=20)

root.mainloop()