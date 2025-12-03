import tkinter as tk
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image, ImageTk
from PIL.Image import Resampling

from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

IMAGE_PATHS = [
    "test1.png",
    "test2.png",
    "test3.png"
]

TARGET_SIZE = (400, 250)


class App:
    def __init__(self) -> None:
        self.root = tk.Tk()
        self.root.title("RGB-анализ изображений")

        self.current_index = 0
        self.current_photo: Optional[ImageTk.PhotoImage] = None

        self.info_label: Optional[tk.Label] = None
        self.canvas: Optional[tk.Canvas] = None

        self.fig: Optional[Figure] = None
        self.ax = None
        self.mpl_canvas: Optional[FigureCanvasTkAgg] = None
        self.bars = None
        self.bar_texts = []

        self.prepared_images: list[dict] = []

        self._build_ui()
        self._prepare_images()

        self.show(0)


    def _build_ui(self) -> None:
        top = tk.Frame(self.root)
        top.pack(anchor="w", fill="x")

        tk.Label(
            top,
            text="Михайлов Дмитрий Андреевич P3306",
            font=("Menlo", 13, "bold"),
        ).pack(side="left", padx=12)

        tk.Button(
            top,
            text="Следующая",
            command=self.next_image,
        ).pack(side="left", padx=6, pady=6)

        self.info_label = tk.Label(
            top,
            text="",
            justify="left",
            font=("Menlo", 11),
        )
        self.info_label.pack(side="left", padx=12)

        self.canvas = tk.Canvas(
            self.root,
            width=TARGET_SIZE[0],
            height=TARGET_SIZE[1],
        )
        self.canvas.pack()

        chart_frame = tk.Frame(self.root)
        chart_frame.pack(fill="x", pady=8)

        self.fig = Figure(figsize=(6, 3), dpi=100)
        self.ax = self.fig.add_subplot(111)

        labels = ["R", "G", "B"]
        initial_data = [0, 0, 0]
        colors = ["red", "green", "blue"]

        self.bars = self.ax.bar(labels, initial_data, color=colors)
        self.ax.set_title("Пиксели по каналам", pad=12)
        self.ax.set_ylabel("Количество пикселей")
        self.ax.set_ylim(0, 1)

        self.bar_texts = []
        for rect in self.bars:
            x = rect.get_x() + rect.get_width() / 2
            text = self.ax.text(x, 0, "0", ha="center", va="bottom")
            self.bar_texts.append(text)

        self.mpl_canvas = FigureCanvasTkAgg(self.fig, master=chart_frame)
        self.mpl_canvas.draw()
        self.mpl_canvas.get_tk_widget().pack(fill="x")


    def _prepare_images(self) -> None:
        self.prepared_images.clear()

        for path_str in IMAGE_PATHS:
            path = Path(path_str)
            if not path.exists():
                print(f"Файл не найден: {path}")
                continue

            src = Image.open(path).convert("RGB")
            view = src.resize(TARGET_SIZE, Resampling.LANCZOS)

            arr = np.asarray(src, dtype=np.uint8)

            r_mean, g_mean, b_mean = arr.mean(axis=(0, 1))

            r = arr[:, :, 0]
            g = arr[:, :, 1]
            b = arr[:, :, 2]

            r_dom = (r > g) & (r > b)
            g_dom = (g > r) & (g > b)
            b_dom = (b > r) & (b > g)

            r_cnt = int(r_dom.sum())
            g_cnt = int(g_dom.sum())
            b_cnt = int(b_dom.sum())

            self.prepared_images.append(
                {
                    "path": str(path),
                    "src": src,
                    "view": view,
                    "mean": (float(r_mean), float(g_mean), float(b_mean)),
                    "counts": (r_cnt, g_cnt, b_cnt),
                }
            )

        if not self.prepared_images:
            raise RuntimeError("Нет ни одного корректного изображения в IMAGE_PATHS")


    def show(self, index: int) -> None:
        img_data = self.prepared_images[index]

        view: Image.Image = img_data["view"]
        self.current_photo = ImageTk.PhotoImage(view)

        assert self.canvas is not None
        if len(self.canvas.find_all()) == 0:
            self.canvas.create_image(0, 0, anchor="nw", image=self.current_photo)
        else:
            self.canvas.itemconfig(1, image=self.current_photo)

        r_m, g_m, b_m = img_data["mean"]
        assert self.info_label is not None
        self.info_label.config(
            text=f"Среднее RGB: ({r_m:.1f}, {g_m:.1f}, {b_m:.1f})"
        )

        r_cnt, g_cnt, b_cnt = img_data["counts"]
        self._update_bar_chart(r_cnt, g_cnt, b_cnt)

    def _update_bar_chart(self, r_cnt: int, g_cnt: int, b_cnt: int) -> None:
        data = [r_cnt, g_cnt, b_cnt]
        max_val = max(data) or 1

        self.ax.set_ylim(0, max_val * 1.15)

        for bar, text, value in zip(self.bars, self.bar_texts, data):
            bar.set_height(value)
            text.set_text(str(value))
            text.set_y(value)

        self.mpl_canvas.draw_idle()


    def next_image(self) -> None:
        self.current_index = (self.current_index + 1) % len(self.prepared_images)
        self.show(self.current_index)

    def run(self) -> None:
        self.root.mainloop()


def main() -> None:
    app = App()
    app.run()


if __name__ == "__main__":
    main()
