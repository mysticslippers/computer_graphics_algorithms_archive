import os
import tkinter as tk
from dataclasses import dataclass
from tkinter import ttk, messagebox, filedialog

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt



@dataclass(frozen=True)
class Params:
    W: float
    H: float
    Wres: int
    Hres: int
    xL: float
    yL: float
    zL: float
    I0: float
    xc: float
    yc: float
    R: float
    fname: str


RANGES = {
    "W": (100.0, 10000.0),
    "H": (100.0, 10000.0),
    "Wres": (200, 800),
    "Hres": (200, 800),
    "xL": (-10000.0, 10000.0),
    "yL": (-10000.0, 10000.0),
    "zL": (100.0, 10000.0),
    "I0": (0.01, 10000.0),
}

def _check_range(name: str, value: float | int):
    lo, hi = RANGES[name]
    if not (lo <= value <= hi):
        raise ValueError(f"{name} вне рекомендуемого диапазона [{lo}..{hi}]: {value}")

def _normalize_square_pixels(W: float, H: float, Wres: int, Hres: int) -> tuple[int, int]:
    px = W / Wres
    py = H / Hres
    if abs(px - py) <= 0.01:
        return Wres, Hres

    step = max(px, py)
    Wres2 = int(round(W / step))
    Hres2 = int(round(H / step))

    Wres2 = max(2, Wres2)
    Hres2 = max(2, Hres2)
    return Wres2, Hres2



def compute_illuminance(p: Params):
    _check_range("W", p.W)
    _check_range("H", p.H)
    _check_range("Wres", p.Wres)
    _check_range("Hres", p.Hres)
    _check_range("xL", p.xL)
    _check_range("yL", p.yL)
    _check_range("zL", p.zL)
    _check_range("I0", p.I0)

    W, H = float(p.W), float(p.H)
    Wres0, Hres0 = int(p.Wres), int(p.Hres)

    Wres, Hres = _normalize_square_pixels(W, H, Wres0, Hres0)

    if not (RANGES["Wres"][0] <= Wres <= RANGES["Wres"][1]) or not (RANGES["Hres"][0] <= Hres <= RANGES["Hres"][1]):
        raise ValueError(
            "После приведения к квадратным пикселям разрешение вышло за 200..800.\n"
            f"Было: Wres={Wres0}, Hres={Hres0}  →  стало: Wres={Wres}, Hres={Hres}\n"
            "Подбери другое Wres/Hres (например, одинаковые значения)."
        )

    x = np.linspace(-W / 2, W / 2, Wres, dtype=np.float32)
    y = np.linspace(-H / 2, H / 2, Hres, dtype=np.float32)

    dx = (x - np.float32(p.xL))[None, :]
    dy = (y - np.float32(p.yL))[:, None]
    z2 = np.float32(p.zL) * np.float32(p.zL)

    r2 = dx * dx + dy * dy + z2
    r4 = r2 * r2

    eps = np.float32(1e-12)
    E_raw = (np.float32(p.I0) * z2) / np.maximum(r4, eps)

    if p.R > 0:
        mx = (x - np.float32(p.xc))[None, :]
        my = (y - np.float32(p.yc))[:, None]
        mask = (mx * mx + my * my) <= np.float32(p.R) * np.float32(p.R)
    else:
        mask = np.ones((Hres, Wres), dtype=bool)

    E = np.where(mask, E_raw, np.float32(0.0))
    Emax = float(E[mask].max()) if mask.any() else 0.0

    if Emax > 0:
        scaled = np.rint((255.0 / Emax) * E).clip(0, 255).astype(np.uint8)
    else:
        scaled = np.zeros((Hres, Wres), dtype=np.uint8)

    extent = (float(x[0]), float(x[-1]), float(y[0]), float(y[-1]))
    return x, y, E_raw, E, scaled, mask, Wres, Hres, Emax, extent


def compute_stats(p: Params, x: np.ndarray, y: np.ndarray, E_raw: np.ndarray, mask: np.ndarray):
    def nearest_idx(arr: np.ndarray, v: float) -> int:
        return int(np.abs(arr - np.float32(v)).argmin())

    pts = [
        ("Center (xc,yc)", (p.xc, p.yc)),
        ("(xc+R,yc)", (p.xc + p.R, p.yc)),
        ("(xc-R,yc)", (p.xc - p.R, p.yc)),
        ("(xc,yc+R)", (p.xc, p.yc + p.R)),
        ("(xc,yc-R)", (p.xc, p.yc - p.R)),
    ]

    point_values = []
    for name, (px, py) in pts:
        ix = nearest_idx(x, px)
        iy = nearest_idx(y, py)
        val = float(E_raw[iy, ix])
        point_values.append((name, px, py, val))

    if mask.any():
        Ein = E_raw[mask]
        Emax = float(Ein.max())
        Emin = float(Ein.min())
        Emean = float(Ein.mean())
    else:
        Emax = Emin = Emean = 0.0

    return point_values, (Emax, Emin, Emean)


def format_stats_text(p: Params, point_values, stats_tuple):
    Emax, Emin, Emean = stats_tuple
    lines = []
    lines.append("Расчёт освещённости (для отчёта)\n")
    lines.append(f"W={p.W} мм, H={p.H} мм, Wres={p.Wres} px, Hres={p.Hres} px")
    lines.append(f"Источник: xL={p.xL} мм, yL={p.yL} мм, zL={p.zL} мм, I0={p.I0} Вт/ср")
    lines.append(f"Круг: xc={p.xc} мм, yc={p.yc} мм, R={p.R} мм\n")

    lines.append("Освещённость в 5 точках (по ближайшему пикселю):")
    for name, px, py, val in point_values:
        lines.append(f"  - {name:14s} @ ({px:.3f}, {py:.3f}) мм : E = {val:.10g}")

    lines.append("\nСтатистики внутри круга:")
    lines.append(f"  - Emax  = {Emax:.10g}")
    lines.append(f"  - Emin  = {Emin:.10g}")
    lines.append(f"  - Emean = {Emean:.10g}")
    return "\n".join(lines)



class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Освещённость от ламбертовского источника")
        self.geometry("760x660")
        self.resizable(True, True)

        self._cache_params: Params | None = None
        self._cache_result = None

        self._build_ui()

    def _build_ui(self):
        frm = ttk.Frame(self, padding=10)
        frm.pack(fill="both", expand=True)

        self.entries: dict[str, tk.StringVar] = {}
        defaults = {
            "W (мм)": 2000, "H (мм)": 2000, "Wres (пкс)": 400, "Hres (пкс)": 400,
            "xL (мм)": 400, "yL (мм)": 200, "zL (мм)": 800, "I0 (Вт/ср)": 250,
            "xc (мм)": 0, "yc (мм)": 0, "R (мм)": 900,
            "Имя файла PNG": "illuminance.png",
        }

        grid = ttk.Frame(frm)
        grid.pack(fill="x", pady=(0, 10))

        for r, (label, val) in enumerate(defaults.items()):
            ttk.Label(grid, text=label).grid(row=r, column=0, sticky="w", padx=(0, 8), pady=4)
            var = tk.StringVar(value=str(val))
            ttk.Entry(grid, textvariable=var, width=22).grid(row=r, column=1, sticky="w")
            self.entries[label] = var

        btns = ttk.Frame(frm)
        btns.pack(fill="x", pady=10)

        ttk.Button(btns, text="Обзор...", command=self._pick_file).pack(side="left", padx=(0, 8))
        ttk.Button(btns, text="Рассчитать и показать", command=self._run).pack(side="left")
        ttk.Button(btns, text="Показать сечения", command=self._show_sections).pack(side="left", padx=8)
        ttk.Button(btns, text="Показать числа для отчёта", command=self._show_stats).pack(side="left", padx=8)
        ttk.Button(btns, text="Сохранить PNG + TXT", command=self._save).pack(side="left", padx=8)
        ttk.Button(btns, text="Сохранить сечения PNG", command=self._save_sections).pack(side="left", padx=8)

        info = ttk.Label(
            frm, foreground="#555",
            text=("Сечение строится через центр заданной области (xc,yc). "
                  "PNG нормируется на максимум освещённости внутри круга.")
        )
        info.pack(fill="x")

    def _pick_file(self):
        path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG", "*.png"), ("All", "*.*")])
        if path:
            self.entries["Имя файла PNG"].set(path)

    def _parse_params(self) -> Params:
        try:
            fname = self.entries["Имя файла PNG"].get().strip() or "illuminance.png"
            return Params(
                W=float(self.entries["W (мм)"].get()),
                H=float(self.entries["H (мм)"].get()),
                Wres=int(self.entries["Wres (пкс)"].get()),
                Hres=int(self.entries["Hres (пкс)"].get()),
                xL=float(self.entries["xL (мм)"].get()),
                yL=float(self.entries["yL (мм)"].get()),
                zL=float(self.entries["zL (мм)"].get()),
                I0=float(self.entries["I0 (Вт/ср)"].get()),
                xc=float(self.entries["xc (мм)"].get()),
                yc=float(self.entries["yc (мм)"].get()),
                R=float(self.entries["R (мм)"].get()),
                fname=fname,
            )
        except Exception as e:
            raise ValueError(f"Ошибка ввода: {e}")

    def _get_result(self):
        p = self._parse_params()
        if p == self._cache_params and self._cache_result is not None:
            return p, self._cache_result

        res = compute_illuminance(p)
        self._cache_params = p
        self._cache_result = res

        *_, Wres_corr, Hres_corr, __, ___ = res
        if Wres_corr != p.Wres or Hres_corr != p.Hres:
            self.entries["Wres (пкс)"].set(str(Wres_corr))
            self.entries["Hres (пкс)"].set(str(Hres_corr))

        return p, res

    def _run(self):
        try:
            p, (x, y, E_raw, E, scaled, mask, Wres, Hres, Emax, extent) = self._get_result()

            plt.figure()
            plt.imshow(scaled, origin="lower", extent=extent, aspect="equal")
            plt.title("Нормированная освещённость (0–255)")
            plt.xlabel("x, мм")
            plt.ylabel("y, мм")
            plt.show()

            denom = Emax if Emax > 0 else float(E_raw.max()) if E_raw.size else 1.0
            y_idx = int(np.abs(y - np.float32(p.yc)).argmin())
            row = (E_raw[y_idx, :] / denom) if denom > 0 else np.zeros_like(x, dtype=np.float32)

            plt.figure()
            plt.plot(x, row)
            plt.title(f"Сечение через центр круга: y = {p.yc} мм")
            plt.xlabel("x, мм")
            plt.ylabel("E (норм., 0–1)")
            plt.grid(True)
            plt.show()
        except Exception as e:
            messagebox.showerror("Ошибка", str(e))

    def _show_sections(self):
        try:
            p, (x, y, E_raw, E, scaled, mask, Wres, Hres, Emax, extent) = self._get_result()
            denom = Emax if Emax > 0 else float(E_raw.max()) if E_raw.size else 1.0

            y_idx = int(np.abs(y - np.float32(p.yc)).argmin())
            row = (E_raw[y_idx, :] / denom) if denom > 0 else np.zeros_like(x, dtype=np.float32)

            plt.figure()
            plt.plot(x, row)
            plt.title(f"Горизонтальное сечение через центр: y = {p.yc} мм")
            plt.xlabel("x, мм")
            plt.ylabel("E (норм., 0–1)")
            plt.grid(True)
            plt.show()

            x_idx = int(np.abs(x - np.float32(p.xc)).argmin())
            col = (E_raw[:, x_idx] / denom) if denom > 0 else np.zeros_like(y, dtype=np.float32)

            plt.figure()
            plt.plot(y, col)
            plt.title(f"Вертикальное сечение через центр: x = {p.xc} мм")
            plt.xlabel("y, мм")
            plt.ylabel("E (норм., 0–1)")
            plt.grid(True)
            plt.show()
        except Exception as e:
            messagebox.showerror("Ошибка", str(e))

    def _show_stats(self):
        try:
            p, (x, y, E_raw, E, scaled, mask, Wres, Hres, Emax, extent) = self._get_result()
            point_values, stats_tuple = compute_stats(p, x, y, E_raw, mask)
            text = format_stats_text(p, point_values, stats_tuple)
            messagebox.showinfo("Числа для отчёта", text)
        except Exception as e:
            messagebox.showerror("Ошибка", str(e))

    def _save(self):
        try:
            p, (x, y, E_raw, E, scaled, mask, Wres, Hres, Emax, extent) = self._get_result()

            Image.fromarray(scaled, mode="L").save(p.fname)

            point_values, stats_tuple = compute_stats(p, x, y, E_raw, mask)
            txt = format_stats_text(p, point_values, stats_tuple)

            base, _ = os.path.splitext(p.fname)
            stats_path = base + "_stats.txt"
            with open(stats_path, "w", encoding="utf-8") as f:
                f.write(txt)

            messagebox.showinfo("Готово", f"Сохранено:\n{p.fname}\n{stats_path}")
        except Exception as e:
            messagebox.showerror("Ошибка", str(e))

    def _save_sections(self):
        try:
            p, (x, y, E_raw, E, scaled, mask, Wres, Hres, Emax, extent) = self._get_result()
            denom = Emax if Emax > 0 else float(E_raw.max()) if E_raw.size else 1.0

            base, _ = os.path.splitext(p.fname)
            hor = base + "_horz.png"
            ver = base + "_vert.png"

            y_idx = int(np.abs(y - np.float32(p.yc)).argmin())
            row = (E_raw[y_idx, :] / denom) if denom > 0 else np.zeros_like(x, dtype=np.float32)

            fig = plt.figure()
            plt.plot(x, row)
            plt.title(f"Горизонтальное сечение через центр: y = {p.yc} мм")
            plt.xlabel("x, мм")
            plt.ylabel("E (норм., 0–1)")
            plt.grid(True)
            plt.savefig(hor, bbox_inches="tight")
            plt.close(fig)

            x_idx = int(np.abs(x - np.float32(p.xc)).argmin())
            col = (E_raw[:, x_idx] / denom) if denom > 0 else np.zeros_like(y, dtype=np.float32)

            fig = plt.figure()
            plt.plot(y, col)
            plt.title(f"Вертикальное сечение через центр: x = {p.xc} мм")
            plt.xlabel("y, мм")
            plt.ylabel("E (норм., 0–1)")
            plt.grid(True)
            plt.savefig(ver, bbox_inches="tight")
            plt.close(fig)

            messagebox.showinfo("Готово", f"Сечения сохранены:\n{hor}\n{ver}")
        except Exception as e:
            messagebox.showerror("Ошибка", str(e))


if __name__ == "__main__":
    App().mainloop()
