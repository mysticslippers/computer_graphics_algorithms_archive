from pathlib import Path
from typing import Optional

import tkinter as tk
from tkinter import filedialog, messagebox

from PIL import (
    Image,
    ImageTk,
    ImageFilter,
    ImageEnhance,
    ImageOps,
)


APP_TITLE = "Filters: Grey, Blur, Contrast, Brightness, Invert - Михайлов Дмитрий Андреевич P3306"

DEFAULT_FILENAME = "test.jpg"

PREVIEW_MAX_WIDTH = 500
PREVIEW_MAX_HEIGHT = 400

RESAMPLING = Image.Resampling.LANCZOS


class ImageEditorApp:
    def __init__(self, master: tk.Tk) -> None:
        self.master = master
        self.master.title(APP_TITLE)
        self.master.resizable(False, False)

        self.orig_image: Optional[Image.Image] = None
        self.processed_image: Optional[Image.Image] = None

        self._tk_orig: Optional[ImageTk.PhotoImage] = None
        self._tk_processed: Optional[ImageTk.PhotoImage] = None

        self.radius_scale: tk.Scale
        self.contrast_scale: tk.Scale
        self.brightness_scale: tk.Scale

        self.left_label: tk.Label
        self.right_label: tk.Label

        self._build_controls()
        self._build_preview_area()

        self._load_default_image()


    def _build_controls(self) -> None:
        top = tk.Frame(self.master)
        top.pack(fill="x", padx=8, pady=8)

        row1 = tk.Frame(top)
        row1.pack(fill="x", pady=(0, 8))

        tk.Button(row1, text="Grey", command=self.make_gray).pack(
            side="left", padx=(0, 8)
        )
        tk.Button(row1, text="Blur", command=self.apply_blur).pack(
            side="left", padx=(0, 16)
        )

        tk.Label(row1, text="Radius:").pack(side="left")
        self.radius_scale = tk.Scale(
            row1,
            from_=0,
            to=20,
            orient="horizontal",
            length=140,
        )
        self.radius_scale.set(2)
        self.radius_scale.pack(side="left", padx=(6, 0))

        row2 = tk.Frame(top)
        row2.pack(fill="x", pady=(0, 8))

        tk.Button(row2, text="Contrast", command=self.apply_contrast).pack(
            side="left", padx=(0, 8)
        )

        tk.Label(row2, text="Factor:").pack(side="left")
        self.contrast_scale = tk.Scale(
            row2,
            from_=0.0,
            to=3.0,
            resolution=0.1,
            orient="horizontal",
            length=120,
        )
        self.contrast_scale.set(1.0)
        self.contrast_scale.pack(side="left", padx=(6, 16))

        tk.Button(row2, text="Brightness", command=self.apply_brightness).pack(
            side="left", padx=(0, 8)
        )

        tk.Label(row2, text="Bright:").pack(side="left")
        self.brightness_scale = tk.Scale(
            row2,
            from_=0.0,
            to=3.0,
            resolution=0.1,
            orient="horizontal",
            length=120,
        )
        self.brightness_scale.set(1.0)
        self.brightness_scale.pack(side="left", padx=(6, 0))

        row3 = tk.Frame(top)
        row3.pack(fill="x")

        tk.Button(
            row3,
            text="Open Image",
            command=self.open_image_dialog,
        ).pack(side="left", padx=(0, 12))

        tk.Button(
            row3,
            text="Invert Colors",
            command=self.invert_colors,
        ).pack(side="left", padx=(0, 12))

        tk.Button(
            row3,
            text="Save PNG",
            command=self.save_image,
        ).pack(side="left")

    def _build_preview_area(self) -> None:
        imgs = tk.Frame(self.master)
        imgs.pack(padx=8, pady=8)

        self.left_label = tk.Label(imgs, text="Исходное", compound="top")
        self.right_label = tk.Label(imgs, text="Результат", compound="top")

        self.left_label.grid(row=0, column=0, padx=8)
        self.right_label.grid(row=0, column=1, padx=8)


    def _load_default_image(self) -> None:
        base_dir = Path(__file__).resolve().parent
        default_path = base_dir / DEFAULT_FILENAME

        if default_path.exists():
            self._load_image(default_path)
        else:
            messagebox.showerror(
                "Нет файла",
                f"Положите рядом файл {DEFAULT_FILENAME} "
                f"или выберите картинку вручную.",
            )

    def open_image_dialog(self) -> None:
        file_path = filedialog.askopenfilename(
            title="Открыть изображение",
            filetypes=[
                ("Images", "*.png;*.jpg;*.jpeg;*.bmp;*.gif;*.tiff"),
                ("All files", "*.*"),
            ],
        )
        if not file_path:
            return

        self._load_image(Path(file_path))

    def _load_image(self, path: Path) -> None:
        try:
            img = Image.open(path).convert("RGBA")
        except Exception as exc:
            messagebox.showerror(
                "Ошибка",
                f"Не удалось открыть файл:\n{path}\n\n{exc}",
            )
            return

        self.orig_image = img
        self.processed_image = None

        self._show_left(self._resize_for_preview(img))
        placeholder = Image.new(
            "RGBA", self._preview_size(img), (240, 240, 240, 255)
        )
        self._show_right(placeholder)


    def save_image(self) -> None:
        if self.processed_image is None:
            messagebox.showwarning(
                "Нет изображения",
                "Сначала примените какой-нибудь фильтр.",
            )
            return

        file_path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"), ("All files", "*.*")],
            title="Сохранить изображение как...",
        )
        if not file_path:
            return

        try:
            self.processed_image.save(file_path, "PNG")
        except Exception as exc:
            messagebox.showerror(
                "Ошибка",
                f"Не удалось сохранить файл:\n{exc}",
            )
        else:
            messagebox.showinfo(
                "Успех",
                f"Изображение сохранено:\n{file_path}",
            )


    @staticmethod
    def _preview_size(image: Image.Image) -> tuple[int, int]:
        w, h = image.size
        ratio = min(
            PREVIEW_MAX_WIDTH / w,
            PREVIEW_MAX_HEIGHT / h,
            1.0,
        )
        return int(w * ratio), int(h * ratio)


    def _resize_for_preview(self, image: Image.Image) -> Image.Image:
        new_size = self._preview_size(image)
        return image.resize(new_size, RESAMPLING)


    def _show_left(self, img: Image.Image) -> None:
        self._tk_orig = ImageTk.PhotoImage(img)
        self.left_label.config(image=self._tk_orig)  # type: ignore[arg-type]


    def _show_right(self, img: Image.Image) -> None:
        self._tk_processed = ImageTk.PhotoImage(img)
        self.right_label.config(image=self._tk_processed) # type: ignore[arg-type]


    def _ensure_image_loaded(self) -> bool:
        if self.orig_image is None:
            messagebox.showwarning(
                "Нет изображения",
                "Сначала загрузите картинку (кнопка Open Image).",
            )
            return False
        return True


    def _update_result(self, img: Image.Image) -> None:
        self.processed_image = img
        self._show_right(self._resize_for_preview(img))


    def make_gray(self) -> None:
        if not self._ensure_image_loaded():
            return

        rgb = self.orig_image.convert("RGB")
        gray_l = ImageOps.grayscale(rgb)
        gray_rgba = gray_l.convert("RGBA")

        if self.orig_image.mode == "RGBA":
            alpha = self.orig_image.getchannel("A")
            gray_rgba.putalpha(alpha)

        self._update_result(gray_rgba)


    def apply_blur(self) -> None:
        if not self._ensure_image_loaded():
            return

        radius = int(self.radius_scale.get())
        blurred = self.orig_image.filter(ImageFilter.GaussianBlur(radius=radius))
        self._update_result(blurred)


    def apply_contrast(self) -> None:
        if not self._ensure_image_loaded():
            return

        factor = float(self.contrast_scale.get())

        rgba = self.orig_image.convert("RGBA")
        rgb = rgba.convert("RGB")
        alpha = rgba.getchannel("A")

        enhanced_rgb = ImageEnhance.Contrast(rgb).enhance(factor)
        r, g, b = enhanced_rgb.split()
        result = Image.merge("RGBA", (r, g, b, alpha))

        self._update_result(result)


    def apply_brightness(self) -> None:
        if not self._ensure_image_loaded():
            return

        factor = float(self.brightness_scale.get())

        rgba = self.orig_image.convert("RGBA")
        rgb = rgba.convert("RGB")
        alpha = rgba.getchannel("A")

        enhanced_rgb = ImageEnhance.Brightness(rgb).enhance(factor)
        r, g, b = enhanced_rgb.split()
        result = Image.merge("RGBA", (r, g, b, alpha))

        self._update_result(result)


    def invert_colors(self) -> None:
        if not self._ensure_image_loaded():
            return

        rgba = self.orig_image.convert("RGBA")
        rgb = rgba.convert("RGB")
        alpha = rgba.getchannel("A")

        inverted_rgb = ImageOps.invert(rgb)
        r, g, b = inverted_rgb.split()
        result = Image.merge("RGBA", (r, g, b, alpha))

        self._update_result(result)


def main() -> None:
    root = tk.Tk()
    ImageEditorApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
