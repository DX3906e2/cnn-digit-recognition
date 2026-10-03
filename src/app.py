"""Tkinter 三阶段可视化窗口（Phase 6.1 Part A：分步交互 + 逐层可视化 + 概率柱状图）。

5 步状态机（每步**只在该步被点入时才计算**，并缓存已算结果）：
  step0 ① 收到图片   原图缩略 / 灰度化+反色+缩放 28×28
  step1 ② 加噪       干净 28×28 / 加噪图
  step2 ③ 去噪       加噪图 / 去噪图 / 5×5 高斯核热图
  step3 ④ CNN 逐层   input→conv1→pool1→conv2→pool2→flatten→softmax 全部中间激活
  step4 ⑤ 识别结果   预测数字 + 置信度 + 10 类概率柱状图

预测管线与训练一致：
  载入图 → data.to_mnist_format() → 加噪(可选) → filters.convolve2d_batch() → to_nchw() → CNN

图片读取用 Tkinter 自带 PhotoImage（PNG/GIF/PPM/PGM），不使用 PIL；不支持格式明确报错。
不引入任何新依赖；参数全部来自 config.py。

运行：      python src/app.py
分步自检：  python src/app.py --selftest <图片> [--no-noise] [--out-prefix outputs/app_selftest]
            → 产出 <前缀>_step0.png ... <前缀>_step4.png 并打印每步数值与耗时
"""
import argparse
import os
import struct
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

import config
import data
import filters
from cnn import CNN
from train import load_model

PARAM_KEYS = ["conv1_W", "conv1_b", "conv2_W", "conv2_b", "fc_W", "fc_b"]

STEP_TITLES = ["① 收到图片", "② 加噪", "③ 去噪", "④ CNN 逐层", "⑤ 识别结果"]
STEP_BUTTONS = ["下一步：加高斯噪声", "下一步：高斯卷积去噪",
                "下一步：送入 CNN", "下一步：输出识别结果", "重新开始"]

# 中文字体（Windows 自带），避免 matplotlib 缺字方块；无该字体时自动回退
import matplotlib  # noqa: E402
matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

_LIB_GRID = 6               # 图片库缩略图网格 6×6 = 36 张
_MNIST_TEST = None          # 懒加载缓存：(X_test, y_test)


def mnist_test_set():
    """懒加载 MNIST 测试集（只读，不修改 data/ 下任何文件）。"""
    global _MNIST_TEST
    if _MNIST_TEST is None:
        _, _, X, y = data.load_mnist(log=lambda *a: None)
        _MNIST_TEST = (X, y)
    return _MNIST_TEST


def mnist_gray_to_rgb(g28):
    """(28,28) float 0~1 → (28,28,3) uint8 RGB。

    量化到 8bit 后走与"存 PNG 再选文件"完全相同的表示，便于两条入口逐位对齐。
    """
    u8 = np.clip(np.round(np.asarray(g28, dtype=np.float64) * 255.0), 0, 255).astype(np.uint8)
    return np.stack([u8, u8, u8], axis=-1)


# --------------------------------------------------------------------------- #
# 图片读取（Tkinter PhotoImage，无 PIL）
# --------------------------------------------------------------------------- #
def photoimage_to_rgb(path):
    """用 Tkinter PhotoImage 读图并转成 (H,W,3) uint8 RGB 数组。

    仅支持 Tk 能解析的格式（PNG/GIF/PPM/PGM 等）；不支持的格式抛 ValueError 并给出
    明确提示。函数内部保证存在 Tk 默认根窗口（外部脚本可直接调用）。
    """
    import tkinter as tk

    # 缺陷修复：外部脚本直接调用时可能还没有默认根窗口
    if tk._default_root is None:
        _r = tk.Tk()
        _r.withdraw()

    ext = os.path.splitext(path)[1].lower()
    try:
        img = tk.PhotoImage(file=path)
    except tk.TclError as e:
        raise ValueError(
            f"无法读取图片“{os.path.basename(path)}”({ext or '未知扩展名'})。"
            f"Tkinter 仅支持 PNG / GIF / PPM / PGM 等格式，请转换后再试。原始错误: {e}"
        )
    w, h = img.width(), img.height()
    arr = np.zeros((h, w, 3), dtype=np.uint8)
    for y in range(h):
        for x in range(w):
            px = img.get(x, y)
            if isinstance(px, str):           # 少数 Tk 版本返回 "r g b"
                px = tuple(int(v) for v in px.split())
            arr[y, x] = px[:3]
    return arr


# --------------------------------------------------------------------------- #
# 剪贴板贴图（纯标准库 ctypes 调 Win32 API 读 CF_DIB；严禁 PIL）
# --------------------------------------------------------------------------- #
def clipboard_image_to_array():
    """读 Windows 剪贴板里的 CF_DIB 位图，返回 (H,W,3) uint8 RGB。

    逐条处理的坑：
      1) OpenClipboard 失败 -> RuntimeError（剪贴板被占用）
      2) 无 CF_DIB    -> ValueError（提示先用 Win+Shift+S 截屏）
      3) BITMAPINFOHEADER 偏移: biSize(0) biWidth(4) biHeight(8) biPlanes(12) biBitCount(14)
      4) biHeight<0 = top-down 不用翻转; >0 需垂直翻转
      5) 每行 4 字节对齐: row_bytes = ((W*bpp+31)//32)*4
      6) 支持 32(BGRA)/24(BGR) -> 转 RGB（B/R 反序）
      7) 像素起点 = ptr + biSize
      8) GlobalUnlock / CloseClipboard 一律放 try/finally
    其它位深（8/16）不在处理范围 -> 明确报错，不静默失败。
    """
    import ctypes
    from ctypes import wintypes

    u32 = ctypes.windll.user32
    k32 = ctypes.windll.kernel32
    CF_DIB = 8

    u32.OpenClipboard.argtypes = [wintypes.HWND]
    u32.OpenClipboard.restype = wintypes.BOOL
    u32.GetClipboardData.argtypes = [wintypes.UINT]
    u32.GetClipboardData.restype = ctypes.c_void_p          # 64 位下必须 c_void_p
    k32.GlobalLock.argtypes = [ctypes.c_void_p]
    k32.GlobalLock.restype = ctypes.c_void_p
    k32.GlobalUnlock.argtypes = [ctypes.c_void_p]

    if not u32.OpenClipboard(None):
        raise RuntimeError("剪贴板被其他程序占用，请稍后重试")
    try:
        if not u32.IsClipboardFormatAvailable(CF_DIB):
            raise ValueError("剪贴板里没有图片。请先用 Win+Shift+S 截屏，再回到窗口按 Ctrl+V")
        handle = u32.GetClipboardData(CF_DIB)
        if not handle:
            raise RuntimeError("读取剪贴板图片句柄失败")
        ptr = k32.GlobalLock(handle)
        if not ptr:
            raise RuntimeError("锁定剪贴板数据失败")
        try:
            hdr = ctypes.string_at(ptr, 16)
            biSize, biWidth, biHeight = struct.unpack_from("<Iii", hdr, 0)
            biPlanes, biBitCount = struct.unpack_from("<HH", hdr, 12)
            if biSize < 40 or biWidth <= 0 or biHeight == 0:
                raise ValueError(f"剪贴板位图头异常: biSize={biSize} W={biWidth} H={biHeight}")
            if biBitCount not in (24, 32):
                raise ValueError(f"暂不支持 {biBitCount} 位剪贴板位图（仅支持 24/32 位），"
                                 f"请改用文件方式载入 PNG")
            bpp = biBitCount // 8
            W, H = biWidth, abs(biHeight)
            row_bytes = ((biWidth * biBitCount + 31) // 32) * 4       # 坑5: 4 字节对齐
            raw = ctypes.string_at(ptr + biSize, row_bytes * H)        # 坑7: 像素起点
            buf = np.frombuffer(raw, dtype=np.uint8).reshape(H, row_bytes)
            buf = buf[:, :W * bpp].reshape(H, W, bpp)
            rgb = buf[:, :, 2::-1] if bpp == 4 else buf[:, :, ::-1]    # 坑6: BGRA/BGR->RGB
            if biHeight > 0:                                          # 坑4: >0 需垂直翻转
                rgb = rgb[::-1]
            return np.ascontiguousarray(rgb)
        finally:
            k32.GlobalUnlock(handle)                                   # 坑8
    finally:
        u32.CloseClipboard()                                           # 坑8


# --------------------------------------------------------------------------- #
# 分步计算（GUI 与 --selftest 共用；每步独立函数，便于计时与验证）
# --------------------------------------------------------------------------- #
def step1_noisy(gray, use_noise, rng):
    """step1: 加噪（关闭时返回 None，表示不加噪）。"""
    if not use_noise:
        return None
    return data.add_gaussian_noise(gray[None], config.NOISE_SIGMA, rng=rng)[0]


def step2_denoise(gray, noisy, use_noise, kernel):
    """step2: 对(加噪后或干净)图做真实高斯卷积去噪。"""
    stage = noisy if use_noise else gray
    return filters.convolve2d_batch(stage[None], kernel)[0]


def step3_layers(cnn, model_input):
    """step3: 逐层前向，返回全部中间激活（A3 指定顺序，不改 layers.py）。"""
    acts = {}
    h = data.to_nchw(model_input[None])
    acts["input"] = h
    h = cnn.conv1.forward(h); acts["conv1"] = h
    h = cnn.relu1.forward(h)
    h = cnn.pool1.forward(h); acts["pool1"] = h
    h = cnn.conv2.forward(h); acts["conv2"] = h
    h = cnn.relu2.forward(h)
    h = cnn.pool2.forward(h); acts["pool2"] = h
    h = cnn.flatten.forward(h); acts["flatten"] = h
    logits = cnn.fc.forward(h); acts["fc"] = logits
    acts["probs"] = cnn.softmax.forward(logits)
    return acts


def step4_predict(acts):
    """step4: 取 softmax 概率 -> 预测类别与置信度。"""
    probs = acts["probs"][0]
    return probs, int(np.argmax(probs)), float(np.max(probs))


def forward_with_acts(cnn, x):
    """A3 版逐层取数（x 为 (1,1,28,28)）；probs 与 cnn.forward(x) 逐位一致。"""
    return step3_layers(cnn, x)


def predict_with_activations(cnn, img28, use_noise, rng):
    """兼容旧接口：按训练管线推理并返回中间量（供外部脚本/回归检查用）。"""
    kernel = filters.gaussian_kernel()
    noisy = step1_noisy(img28, use_noise, rng)
    denoised = step2_denoise(img28, noisy, use_noise, kernel)
    acts = step3_layers(cnn, denoised)
    probs, pred, conf = step4_predict(acts)
    return {"noisy": noisy, "denoised": denoised, "model_input": denoised,
            "conv1_map": acts["conv1"][0], "conv2_map": acts["conv2"][0],
            "probs": probs, "pred": pred, "conf": conf, "kernel": kernel,
            "acts": acts}


# --------------------------------------------------------------------------- #
# 绘图（不依赖 Tk，GUI 与 --selftest 共用）
# --------------------------------------------------------------------------- #
def _hide(ax):
    ax.set_xticks([]); ax.set_yticks([])


def _title(ax, text, fontsize):
    ax.set_title(text, fontsize=fontsize, loc="left")


def draw_step0(fig, rgb, gray):
    fig.clear()
    ax = fig.add_subplot(1, 2, 1)
    ax.imshow(rgb); _title(ax, f"原图 RGB {rgb.shape}", 9); _hide(ax)
    rgb_gray = (0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]) / 255.0
    inv = bool(rgb_gray.mean() > 0.5)
    ax = fig.add_subplot(1, 2, 2)
    ax.imshow(gray, cmap="gray", vmin=0, vmax=1)
    _title(ax, f"灰度化 28×28  (auto_inverted={inv})", 9); _hide(ax)
    fig.suptitle(f"① 收到图片   raw_gray_mean={rgb_gray.mean():.4f}   "
                 f"auto_inverted={inv}   gray_mean={gray.mean():.4f}", fontsize=10)
    return {"raw_shape": rgb.shape, "raw_gray_mean": float(rgb_gray.mean()),
            "auto_inverted": inv, "gray_mean": float(gray.mean())}


def draw_step1(fig, gray, noisy, use_noise):
    fig.clear()
    ax = fig.add_subplot(1, 2, 1)
    ax.imshow(gray, cmap="gray", vmin=0, vmax=1)
    _title(ax, f"干净 28×28  std={gray.std():.4f}", 9); _hide(ax)
    info = {"clean_std": float(gray.std())}
    ax = fig.add_subplot(1, 2, 2)
    if use_noise:
        ax.imshow(noisy, cmap="gray", vmin=0, vmax=1)
        mse = float(np.mean((noisy - gray) ** 2))
        _title(ax, f"加噪  sigma={config.NOISE_SIGMA}  std={noisy.std():.4f}", 9)
        info.update({"noisy_std": float(noisy.std()), "noisy_mse": mse})
        sub = f"sigma={config.NOISE_SIGMA}   noisy_std={noisy.std():.4f}   MSE(noisy,clean)={mse:.4f}"
    else:
        ax.imshow(gray, cmap="gray", vmin=0, vmax=1)
        _title(ax, "加噪：已关闭", 9)
        info.update({"noisy_std": None, "noisy_mse": 0.0})
        sub = "加噪：已关闭（右图=干净图, MSE=0）"
    _hide(ax)
    fig.suptitle(f"② 加噪   {sub}", fontsize=10)
    return info


def draw_step2(fig, gray, noisy, denoised, kernel, use_noise):
    fig.clear()
    left = noisy if use_noise else gray
    ax = fig.add_subplot(1, 3, 1)
    ax.imshow(left, cmap="gray", vmin=0, vmax=1)
    _title(ax, "加噪图" if use_noise else "干净图(未加噪)", 9); _hide(ax)
    ax = fig.add_subplot(1, 3, 2)
    ax.imshow(denoised, cmap="gray", vmin=0, vmax=1)
    mse = float(np.mean((denoised - gray) ** 2))
    _title(ax, f"去噪后  std={denoised.std():.4f}", 9); _hide(ax)
    ax = fig.add_subplot(1, 3, 3)
    im = ax.imshow(kernel, cmap="viridis")
    _title(ax, f"高斯核 {config.GAUSS_K}×{config.GAUSS_K}  sum={kernel.sum():.6f}", 9)
    _hide(ax)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(f"③ 去噪   去噪后 std={denoised.std():.4f}   "
                 f"MSE(denoised,clean)={mse:.4f}   核和={kernel.sum():.6f}", fontsize=10)
    return {"denoised_std": float(denoised.std()), "denoised_mse": mse,
            "kernel_sum": float(kernel.sum())}


def draw_step3(fig, acts):
    """④ CNN 逐层：input→conv1→pool1→conv2→pool2→flatten→softmax 全量激活。"""
    fig.clear()
    outer = fig.add_gridspec(13, 4, hspace=0.55, wspace=0.12)
    info = {"layers": {}}

    ax = fig.add_subplot(outer[0, 0])
    ax.imshow(acts["input"][0, 0], cmap="gray", vmin=0, vmax=1)
    _title(ax, f"input {acts['input'].shape[1:]}", 7); _hide(ax)

    ax = fig.add_subplot(outer[0, 1])
    fl = acts["flatten"][0].reshape(config.IMG_SIZE, config.IMG_SIZE)
    ax.imshow(fl, cmap="viridis")
    _title(ax, f"flatten {acts['flatten'].shape[1]}→28×28", 7); _hide(ax)
    info["layers"]["flatten"] = _layer_stat(acts["flatten"])

    ax = fig.add_subplot(outer[0, 2:4])
    draw_prob_bars(ax, acts["probs"][0], fontsize=6)
    info["layers"]["probs"] = _layer_stat(acts["probs"])

    groups = [("conv1", 1, 3, 2, 4), ("pool1", 3, 5, 2, 4),
              ("conv2", 5, 9, 4, 4), ("pool2", 9, 13, 4, 4)]
    for name, r0, r1, rows, cols in groups:
        f = acts[name][0]                      # (C,H,W)
        C, H, W = f.shape
        info["layers"][name] = _layer_stat(f)
        sub = outer[r0:r1, :].subgridspec(rows, cols, hspace=0.6, wspace=0.10)
        for k in range(min(C, rows * cols)):
            axk = fig.add_subplot(sub[k // cols, k % cols])
            axk.imshow(f[k], cmap="viridis"); _hide(axk)
            head = (f"{name} {C}×{H}×{W} min={f.min():.2f} max={f.max():.2f}\n"
                    if k == 0 else "")
            axk.set_title(f"{head}ch{k} std={f[k].std():.2f}", fontsize=5.5)

    fig.suptitle("④ CNN 逐层中间激活（viridis，标题含每通道 std 与层 min/max）", fontsize=10)
    return info


def _layer_stat(a):
    a = np.asarray(a)
    return {"shape": tuple(a.shape), "std": float(a.std()),
            "min": float(a.min()), "max": float(a.max()),
            "nonzero": int(np.count_nonzero(a))}


def draw_prob_bars(ax, probs, fontsize=8):
    """0~9 概率柱状图：最高柱高亮、柱顶标 4 位小数、标题含 Top-2 与低置信度告警。"""
    n = len(probs)
    am = int(np.argmax(probs))
    colors = ["tab:red" if i == am else "tab:gray" for i in range(n)]
    bars = ax.bar(range(n), probs, color=colors)
    for b, p in zip(bars, probs):
        ax.text(b.get_x() + b.get_width() / 2, p, f"{p:.4f}",
                ha="center", va="bottom", fontsize=fontsize)
    ax.set_xticks(range(n))
    ax.set_ylim(0, 1.12)
    ax.set_xlabel("class", fontsize=fontsize + 1)
    ax.set_ylabel("probability", fontsize=fontsize + 1)
    order = np.argsort(probs)[::-1]
    t1, t2 = int(order[0]), int(order[1])
    title = (f"预测：{t1}   置信度：{probs[t1]:.4f}   "
             f"Top2: {t1} ({probs[t1]:.4f}) / {t2} ({probs[t2]:.4f})")
    if probs[t1] < 0.6:
        title += "  ⚠ 低置信度"
    ax.set_title(title, fontsize=fontsize + 2)


def draw_step4(fig, gray, probs, pred, conf):
    fig.clear()
    ax = fig.add_subplot(1, 2, 1)
    ax.imshow(gray, cmap="gray", vmin=0, vmax=1)
    ax.text(0.5, -0.12, f"预测 {pred}    置信度 {conf:.4f}", transform=ax.transAxes,
            ha="center", va="top", fontsize=16)
    _hide(ax)
    ax = fig.add_subplot(1, 2, 2)
    draw_prob_bars(ax, probs, fontsize=8)
    fig.suptitle(f"⑤ 识别结果：数字 {pred}   置信度 {conf:.4f}", fontsize=13)
    return {"pred": pred, "conf": conf, "probs": [float(p) for p in probs]}


# --------------------------------------------------------------------------- #
# GUI
# --------------------------------------------------------------------------- #
class App:
    def __init__(self, root):
        import tkinter as tk
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        import tkinter.filedialog as filedialog
        import tkinter.messagebox as messagebox

        self.tk, self.filedialog, self.messagebox = tk, filedialog, messagebox
        self.root = root
        root.title("MNIST 五步可视化（Phase 6.1）")

        self.cnn = CNN()
        load_model(self.cnn, config.MODEL_PATH)
        self.kernel = filters.gaussian_kernel()
        self.rng = np.random.default_rng(config.SEED)

        self.use_noise = tk.BooleanVar(value=True)
        self.step = -1
        self.rgb = self.gray = None
        self.path = None
        self.step_ms = {}
        self._reset_cache()

        # ---- 左侧控制区 ----
        left = tk.Frame(root)
        left.pack(side=tk.LEFT, fill=tk.Y, padx=8, pady=8)

        tk.Label(left, text='请先"选择图片"或"从剪贴板粘贴"（Win+Shift+S 截屏后 Ctrl+V）',
                 font=("Microsoft YaHei", 9), fg="gray", wraplength=250,
                 justify="left").pack(anchor="w", pady=(0, 6))
        tk.Button(left, text="选择图片 (PNG/GIF)", width=26,
                  command=self.choose).pack(pady=2)
        tk.Button(left, text="从剪贴板粘贴 (Ctrl+V)", width=26,
                  command=self.paste).pack(pady=2)
        tk.Button(left, text="从数据集选择…", width=26,
                  command=self.open_library).pack(pady=2)
        self._lib = None
        root.bind("<Control-v>", self.paste)
        root.bind("<Control-V>", self.paste)
        tk.Checkbutton(left, text=f"加噪 (sigma={config.NOISE_SIGMA})",
                       variable=self.use_noise, command=self.on_toggle_noise).pack(anchor="w")
        self.lbl_model = tk.Label(left, text=f"model: {os.path.basename(config.MODEL_PATH)}",
                                  font=("Consolas", 9), anchor="w")
        self.lbl_model.pack(anchor="w", pady=(6, 0))

        self.btn_next = tk.Button(left, text="下一步", width=26, command=self.next)
        self.btn_next.pack(pady=(10, 2))
        self.btn_prev = tk.Button(left, text="上一步", width=26, command=self.prev)
        self.btn_prev.pack(pady=2)
        tk.Button(left, text="重新开始", width=26, command=self.restart).pack(pady=2)

        self.lbl_step = tk.Label(left, text="当前步骤 -/5", font=("Microsoft YaHei", 11),
                                 anchor="w")
        self.lbl_step.pack(anchor="w", pady=(10, 0))
        self.lbl_detail = tk.Label(left, text="", font=("Consolas", 8), justify="left",
                                   anchor="w", wraplength=260)
        self.lbl_detail.pack(anchor="w")
        self.lbl_status = tk.Label(left, text="", font=("Microsoft YaHei", 8), fg="#0a6",
                                   justify="left", anchor="w", wraplength=260)
        self.lbl_status.pack(anchor="w", pady=(8, 0))

        # ---- 右侧画布 ----
        self.fig = Figure(figsize=(10, 8), dpi=100, layout="constrained")
        self.canvas = FigureCanvasTkAgg(self.fig, master=root)
        self.canvas.get_tk_widget().pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self._draw_blank()
        self._update_controls()

    # ---- 缓存与状态 ----
    def _reset_cache(self):
        self._noisy = None
        self._denoised = None
        self._acts = None
        self._probs = None

    def restart(self):
        self.step = -1
        self.rgb = self.gray = None
        self.path = None
        self.step_ms = {}
        self._reset_cache()
        self.lbl_status.config(text="")
        self._draw_blank()
        self._update_controls()

    def on_toggle_noise(self):
        self._reset_cache()          # 加噪开关影响 step1/2 及后续全部计算
        self.step_ms = {}
        if self.gray is not None and self.step >= 0:
            self.go(self.step)       # 重算当前步
        else:
            self._update_controls()

    def choose(self):
        path = self.filedialog.askopenfilename(
            title="选择图片",
            filetypes=[("图片 (PNG/GIF/PPM/PGM)", "*.png *.gif *.ppm *.pgm"),
                       ("所有文件", "*.*")])
        if not path:
            return
        try:
            rgb = photoimage_to_rgb(path)
        except ValueError as e:
            self.messagebox.showerror("读取失败", str(e))
            return
        self._load_rgb(rgb, os.path.basename(path))

    def paste(self, event=None):
        """从剪贴板读图（Ctrl+V 或按钮）。成功后回到 step 0 并重走流程。"""
        try:
            rgb = clipboard_image_to_array()
        except (RuntimeError, ValueError) as e:
            self.messagebox.showerror("粘贴失败", str(e))
            self.lbl_status.config(text=f"粘贴失败：{e}")
            return "break"
        h, w = rgb.shape[0], rgb.shape[1]
        self._load_rgb(rgb, f"剪贴板 {w}×{h} 截图")
        return "break"

    def _load_rgb(self, rgb, source):
        """统一入口：文件 / 剪贴板 / 数据集三条路径都走这里（同一条 to_mnist_format 管线）。"""
        self.rgb = rgb
        self.gray = data.to_mnist_format(rgb)      # 复用既有管线，不另写灰度/缩放
        self.path = source
        self.step_ms = {}
        self._reset_cache()
        self.lbl_status.config(text=f"已从 {source} 读取，并回到 ① 收到图片")
        self.go(0)

    def open_library(self):
        """打开 MNIST 图片库窗口（只挑图，不做任何预处理/预测）。"""
        if self._lib is not None and self._lib.winfo_exists():
            self._lib.lift()
            return
        self._lib = LibraryWindow(self)

    def load_from_dataset(self, idx, label):
        """图片库回调：把选中的 MNIST 图交给统一入口（不单开捷径）。"""
        X, _ = mnist_test_set()
        rgb = mnist_gray_to_rgb(X[idx])
        self._load_rgb(rgb, f"MNIST test #{idx} (label {label})")

    # ---- 分步推进（step N 的计算只在此刻发生）----
    def _compute(self, step):
        """计算到 step 为止所需的量；已缓存的层不重算。"""
        if step >= 1 and self.use_noise.get() and self._noisy is None:
            self._noisy = step1_noisy(self.gray, True, self.rng)
        if step >= 2 and self._denoised is None:
            self._denoised = step2_denoise(self.gray, self._noisy, self.use_noise.get(), self.kernel)
        if step >= 3 and self._acts is None:
            self._acts = step3_layers(self.cnn, self._denoised)
        if step >= 4 and self._probs is None:
            if self._acts is None:
                self._acts = step3_layers(self.cnn, self._denoised)
            self._probs = step4_predict(self._acts)

    def go(self, step):
        if self.gray is None:
            return
        step = max(0, min(4, step))
        t0 = time.perf_counter()
        self._compute(step)
        self.step_ms[step] = (time.perf_counter() - t0) * 1000.0
        self.step = step
        self._draw()
        self._update_controls()

    def next(self):
        if self.step < 0:
            return
        if self.step == 4:
            self.restart()
        else:
            self.go(self.step + 1)

    def prev(self):
        if self.step > 0:
            self.go(self.step - 1)

    # ---- 绘制 ----
    def _draw_blank(self):
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        ax.text(0.5, 0.5, "Click 'Select image' to load a PNG / GIF", ha="center", va="center")
        ax.axis("off")
        self.canvas.draw()

    def _draw(self):
        s = self.step
        if s == 0:
            info = draw_step0(self.fig, self.rgb, self.gray)
        elif s == 1:
            info = draw_step1(self.fig, self.gray, self._noisy, self.use_noise.get())
        elif s == 2:
            info = draw_step2(self.fig, self.gray, self._noisy, self._denoised,
                              self.kernel, self.use_noise.get())
        elif s == 3:
            info = draw_step3(self.fig, self._acts)
        else:
            probs, pred, conf = step4_predict(self._acts)
            info = draw_step4(self.fig, self.gray, probs, pred, conf)
        self.canvas.draw()

        self.lbl_step.config(text=f"当前步骤 {s + 1}/5  {STEP_TITLES[s]}")
        self.lbl_detail.config(text=self._detail_text(s, info))

    def _detail_text(self, s, info):
        ms = self.step_ms.get(s, 0.0)
        lines = [f"本步耗时: {ms:.1f} ms"]
        if s == 3:
            for k in ("conv1", "pool1", "conv2", "pool2", "flatten"):
                st = info["layers"][k]
                lines.append(f"{k:8s} {st['shape']} std={st['std']:.3f} "
                             f"[{st['min']:.2f},{st['max']:.2f}]")
        elif s == 4:
            lines.append(f"pred={info['pred']} conf={info['conf']:.4f}")
        elif s == 2:
            lines.append(f"denoised_mse={info['denoised_mse']:.4f}")
        elif s == 1 and info.get("noisy_std") is not None:
            lines.append(f"noisy_std={info['noisy_std']:.4f} mse={info['noisy_mse']:.4f}")
        return "\n".join(lines)

    def _update_controls(self):
        if self.step < 0:
            self.btn_next.config(text="下一步", state="disabled")
            self.btn_prev.config(state="disabled")
        else:
            self.btn_next.config(text=STEP_BUTTONS[self.step], state="normal")
            self.btn_prev.config(state="normal" if self.step > 0 else "disabled")


# --------------------------------------------------------------------------- #
# MNIST 图片库窗口（只挑图；处理全交给主窗口五步管线）
# --------------------------------------------------------------------------- #
class LibraryWindow:
    """按标签检索 MNIST 测试集，6×6 缩略图网格，点选后回主窗口走五步。

    本窗口**不加噪、不去噪、不预测**（红线 5），只调用 app.load_from_dataset()。
    """

    def __init__(self, app):
        import tkinter as tk
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

        self.app = app
        self.tk = tk
        self.X, self.y = mnist_test_set()
        self.label = None                     # None = 全部
        self.cells = {}                       # ax -> 测试集索引
        self.top = tk.Toplevel(app.root)
        self.top.title("MNIST 图片库（点缩略图载入）")

        bar = tk.Frame(self.top)
        bar.pack(side=tk.TOP, fill=tk.X, padx=6, pady=4)
        tk.Label(bar, text="标签:", font=("Microsoft YaHei", 9)).pack(side=tk.LEFT)
        tk.Button(bar, text="全部", width=4, command=lambda: self.set_label(None)).pack(side=tk.LEFT, padx=1)
        for d in range(config.NUM_CLASSES):
            tk.Button(bar, text=str(d), width=3,
                      command=lambda d=d: self.set_label(d)).pack(side=tk.LEFT, padx=1)
        tk.Button(bar, text="换一批", width=6, command=self.refresh).pack(side=tk.LEFT, padx=8)
        self.lbl = tk.Label(bar, text="", font=("Microsoft YaHei", 9))
        self.lbl.pack(side=tk.LEFT, padx=8)

        self.fig = Figure(figsize=(7.5, 7.5), dpi=100, layout="constrained")
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.top)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.canvas.mpl_connect("button_press_event", self.on_click)

        self.refresh()

    def set_label(self, d):
        self.label = d
        self.refresh()

    def _pool(self):
        if self.label is None:
            return np.arange(len(self.y))
        return np.where(self.y == self.label)[0]

    def refresh(self):
        pool = self._pool()
        k = min(_LIB_GRID * _LIB_GRID, len(pool))
        rng = np.random.default_rng()                 # 「换一批」每次不同
        pick = rng.choice(pool, size=k, replace=False)
        self.fig.clear()
        self.cells = {}
        axes = self.fig.subplots(_LIB_GRID, _LIB_GRID)
        for c in range(_LIB_GRID * _LIB_GRID):
            ax = axes[c // _LIB_GRID, c % _LIB_GRID]
            ax.set_xticks([]); ax.set_yticks([])
            if c < k:
                idx = int(pick[c])
                ax.imshow(self.X[idx], cmap="gray", vmin=0.0, vmax=1.0)
                ax.set_title(f"#{idx}\nL{int(self.y[idx])}", fontsize=5)
                self.cells[ax] = idx
            else:
                ax.axis("off")
        tag = "全部" if self.label is None else f"标签 {self.label}"
        self.lbl.config(text=f"{tag}：候选 {len(pool)} 张，显示 {k} 张（点缩略图载入）")
        self.canvas.draw()

    def on_click(self, event):
        if event.inaxes is None:
            return
        idx = self.cells.get(event.inaxes)
        if idx is None:
            return
        label = int(self.y[idx])
        self.top.destroy()
        self.app._lib = None
        self.app.load_from_dataset(idx, label)


# --------------------------------------------------------------------------- #
# 分步自检（无窗口；导出 5 张面板图 + 打印每步数值与耗时）
# --------------------------------------------------------------------------- #
def selftest(path=None, use_noise=True, prefix=None, dataset=None, label=None):
    """分步自检：严格按 step0→4 逐步计算并出图，打印每步数值与**该步真实耗时**。

    每步的计算发生在该步内，不在开头一次性算完（便于验证"分步是真的"）。
    两种图源（都走同一条 to_mnist_format 管线）：
      - path   : 本地图片路径
      - dataset: MNIST 测试集第 dataset 张（若给 label，则表示"标签 label 下的第 dataset 张"）
    """
    matplotlib.use("Agg")
    from matplotlib.figure import Figure

    if prefix is None:
        prefix = os.path.join(config.OUTPUT_DIR, "app_selftest")
    cnn = CNN()
    load_model(cnn, config.MODEL_PATH)
    kernel = filters.gaussian_kernel()
    rng = np.random.default_rng(config.SEED)

    out_dir = os.path.dirname(os.path.abspath(prefix))
    os.makedirs(out_dir, exist_ok=True)

    def run(figsize, drawfn, *a, **kw):
        t = time.perf_counter()
        fig = Figure(figsize=figsize, dpi=100, layout="constrained")
        info = drawfn(fig, *a, **kw)
        return info, (time.perf_counter() - t) * 1000.0, fig

    # step0：载入 + 灰度化（两种图源二选一）
    true_label = None
    t0 = time.perf_counter()
    if path is not None:
        rgb = photoimage_to_rgb(path)
        source = path
    else:
        X, y = mnist_test_set()
        n = 0 if dataset is None else int(dataset)
        if label is None:
            idx = int(n)
        else:
            hits = np.where(y == int(label))[0]
            if len(hits) == 0:
                raise ValueError(f"测试集中没有标签 {label}")
            idx = int(hits[n % len(hits)])
        true_label = int(y[idx])
        rgb = mnist_gray_to_rgb(X[idx])
        source = f"MNIST test #{idx} (label {true_label})"
    gray = data.to_mnist_format(rgb)
    ms0 = (time.perf_counter() - t0) * 1000.0
    i0, _, fig = run((9, 4.5), draw_step0, rgb, gray)
    fig.savefig(prefix + "_step0.png", dpi=110)
    print(f"[selftest] source     : {source}")
    if true_label is not None:
        print(f"[selftest] true label : {true_label}")
    print(f"[selftest] step0 ①     : raw{tuple(i0['raw_shape'])} raw_gray_mean={i0['raw_gray_mean']:.4f} "
          f"auto_inverted={i0['auto_inverted']} gray_mean={i0['gray_mean']:.4f}   ({ms0:.2f} ms)")

    # step1：加噪（只在此刻算）
    t1 = time.perf_counter()
    noisy = step1_noisy(gray, use_noise, rng)
    ms1 = (time.perf_counter() - t1) * 1000.0
    i1, _, fig = run((9, 4.5), draw_step1, gray, noisy, use_noise)
    fig.savefig(prefix + "_step1.png", dpi=110)
    if use_noise:
        print(f"[selftest] step1 ②     : sigma={config.NOISE_SIGMA} "
              f"noisy_std={i1['noisy_std']:.4f} MSE(noisy,clean)={i1['noisy_mse']:.4f}   ({ms1:.2f} ms)")
    else:
        print(f"[selftest] step1 ②     : 加噪=关闭 (noisy=None)   ({ms1:.2f} ms)")

    # step2：去噪（只在此刻算）
    t2 = time.perf_counter()
    den = step2_denoise(gray, noisy, use_noise, kernel)
    ms2 = (time.perf_counter() - t2) * 1000.0
    i2, _, fig = run((11, 4.5), draw_step2, gray, noisy, den, kernel, use_noise)
    fig.savefig(prefix + "_step2.png", dpi=110)
    print(f"[selftest] step2 ③     : denoised_std={i2['denoised_std']:.4f} "
          f"MSE(denoised,clean)={i2['denoised_mse']:.4f} kernel_sum={i2['kernel_sum']:.6f}   ({ms2:.2f} ms)")

    # step3：逐层前向（只在此刻算）
    t3 = time.perf_counter()
    acts = step3_layers(cnn, den)
    ms3 = (time.perf_counter() - t3) * 1000.0
    i3, ms3_draw, fig = run((12, 16), draw_step3, acts)
    fig.savefig(prefix + "_step3.png", dpi=110)
    for k in ("conv1", "pool1", "conv2", "pool2", "flatten"):
        st = i3["layers"][k]
        print(f"[selftest] step3 ④ {k:8s}: shape={st['shape']} std={st['std']:.4f} "
              f"min={st['min']:.3f} max={st['max']:.3f} nonzero={st['nonzero']}")
    print(f"[selftest] step3 ④     : 逐层取数 {ms3:.2f} ms (绘图 {ms3_draw:.2f} ms)")

    # step4：识别结果
    t4 = time.perf_counter()
    probs, pred, conf = step4_predict(acts)
    ms4 = (time.perf_counter() - t4) * 1000.0
    i4, _, fig = run((11, 4.5), draw_step4, gray, probs, pred, conf)
    fig.savefig(prefix + "_step4.png", dpi=110)
    extra = "" if true_label is None else f"   true={true_label}   一致={pred == true_label}"
    print(f"[selftest] step4 ⑤     : pred={pred} conf={conf:.6f}{extra}   ({ms4:.2f} ms)")
    print(f"[selftest] probs       : {[round(float(p), 4) for p in probs]}")
    print(f"[selftest] panels      : {prefix}_step0.png ... _step4.png")
    return {"pred": pred, "conf": conf, "probs": probs, "acts": acts, "gray": gray,
            "noisy": noisy, "denoised": den, "ms": [ms0, ms1, ms2, ms3, ms4]}


def main():
    ap = argparse.ArgumentParser(description="MNIST 五步可视化窗口（Phase 6.1 Part A）")
    ap.add_argument("--model", type=str, default=None)
    ap.add_argument("--selftest", type=str, default=None, help="无窗口自检：本地图片路径")
    ap.add_argument("--dataset", type=int, default=None,
                    help="无窗口自检：改用 MNIST 测试集第 N 张（与 --label 联用表示该标签下第 N 张）")
    ap.add_argument("--label", type=int, default=None, help="配合 --dataset：限定真实标签 0~9")
    ap.add_argument("--no-noise", action="store_true", help="自检时关闭加噪")
    ap.add_argument("--out-prefix", type=str,
                    default=os.path.join(config.OUTPUT_DIR, "app_selftest"),
                    help="自检面板图前缀，默认 outputs/app_selftest")
    ap.add_argument("--out", type=str, default=None, help="(兼容旧参) 等价于 --out-prefix，去掉 .png")
    args = ap.parse_args()
    if args.model:
        config.MODEL_PATH = args.model

    if args.selftest or args.dataset is not None:
        prefix = args.out_prefix
        if args.out:
            prefix = args.out[:-4] if args.out.lower().endswith(".png") else args.out
        selftest(path=args.selftest, use_noise=not args.no_noise, prefix=prefix,
                 dataset=args.dataset, label=args.label)
        return

    import tkinter as tk
    root = tk.Tk()
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
