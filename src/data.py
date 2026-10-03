import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

"""MNIST 本地读取 + 手写 IDX 解析 + 高斯噪声注入（Phase 2，决策 D13/D15）。

依赖：仅 numpy + 标准库（urllib / gzip / struct / os）。
禁止：sklearn / scipy / cv2 / torch / PIL（不在依赖清单）。

入口：
  load_mnist()      -> (X_train, y_train, X_test, y_test)
  add_gaussian_noise(X, sigma=None)
  to_mnist_format(img)   -> (28, 28) float64 0~1（不负责文件 IO）
所有可调参数来自 config.py，本文件不出现魔法数字。
"""

import gzip
import os
import struct
import urllib.request
from pathlib import Path

import numpy as np

import config


# ---- MNIST 镜像兜底顺序（决策 D13：手写 IDX 下载，内置镜像兜底）----
_MIRRORS = (
    "https://storage.googleapis.com/cvdf-datasets/mnist/",
    "https://ossci-datasets.s3.amazonaws.com/mnist/",
    "http://yann.lecun.com/exdb/mnist/",  # 原站，最后尝试
)

# 4 个标准文件（.gz 压缩的 IDX）
_FILES = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images": "t10k-images-idx3-ubyte.gz",
    "test_labels": "t10k-labels-idx1-ubyte.gz",
}

# IDX magic number
_IMG_MAGIC = 2051   # 0x00000803
_LBL_MAGIC = 2049   # 0x00000801

# ---- to_mnist_format 预处理参数（Phase 8）----
_FG_THRESHOLD = 0.15        # 前景判定阈值（反色后数字为亮）
_CROP_MARGIN = 4            # 裁剪居中后四周留白像素
_CROP_SKIP_RATIO = 0.85     # bbox 覆盖画面 >= 此比例 -> 已占满，跳过裁剪
_OTSU_BINS = 256            # Otsu 直方图箱数
_OTSU_WARN_HI = 0.9         # 阈值极端上界（超过则警告）
_OTSU_WARN_LO = 0.1         # 阈值极端下界（低于则警告）


def _resolve_dir(path):
    """相对路径以仓库根解析；返回绝对路径目录。"""
    p = config.ROOT / path if not os.path.isabs(path) else Path(path)
    return Path(p)


def _download_one(filename, mirrors=None, dest_dir=None, log=print):
    """下载单个文件，逐个镜像尝试；目标已存在则直接命中缓存。

    返回 (dest_path, "downloaded" | "cached")。
    """
    mirrors = mirrors or _MIRRORS
    dest_root = _resolve_dir(dest_dir) if dest_dir else _resolve_dir(config.MNIST_DIR)
    os.makedirs(dest_root, exist_ok=True)
    dest = dest_root / filename

    if dest.exists():
        log(f"  [cached] {filename}  (mtime={dest.stat().st_mtime:.0f})")
        return dest, "cached"

    last_err = None
    for url_base in mirrors:
        url = url_base.rstrip("/") + "/" + filename
        try:
            log(f"  [try]    {filename} <- {url}")
            urllib.request.urlretrieve(url, dest)
            log(f"  [ok]     {filename} <- {url}")
            return dest, "downloaded"
        except Exception as e:  # 任一镜像失败，切换到下一个
            last_err = e
            log(f"  [fail]   {filename} <- {url}  ({type(e).__name__}: {e})")
    raise RuntimeError(f"all mirrors failed for {filename}: {last_err}")


def _parse_idx_images(path):
    """手写解析 IDX 图像文件（大端）。返回 (N, 28, 28) float64 / 255。"""
    with gzip.open(path, "rb") as f:
        magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
    if magic != _IMG_MAGIC:
        raise ValueError(f"图像 magic 不匹配：期望 {_IMG_MAGIC}，实际 {magic}（{path}）")
    if (rows, cols) != (config.IMG_SIZE, config.IMG_SIZE):
        raise ValueError(f"图像尺寸不匹配：期望 {config.IMG_SIZE}x{config.IMG_SIZE}，实际 {rows}x{cols}")
    with gzip.open(path, "rb") as f:
        f.seek(16)
        buf = f.read()
    data = np.frombuffer(buf, dtype=np.uint8)
    if data.size != num * rows * cols:
        raise ValueError(f"图像像素数不匹配：期望 {num*rows*cols}，实际 {data.size}")
    return data.reshape(num, rows, cols).astype(np.float64) / 255.0


def _parse_idx_labels(path):
    """手写解析 IDX 标签文件（大端）。返回 (N,) int64。"""
    with gzip.open(path, "rb") as f:
        magic, num = struct.unpack(">II", f.read(8))
    if magic != _LBL_MAGIC:
        raise ValueError(f"标签 magic 不匹配：期望 {_LBL_MAGIC}，实际 {magic}（{path}）")
    with gzip.open(path, "rb") as f:
        f.seek(8)
        buf = f.read()
    data = np.frombuffer(buf, dtype=np.uint8)
    if data.size != num:
        raise ValueError(f"标签数不匹配：期望 {num}，实际 {data.size}")
    return data.reshape(num).astype(np.int64)


def load_mnist(n_train=None, log=print):
    """加载 MNIST。缺失文件时按镜像顺序下载；返回 (X_train, y_train, X_test, y_test)。

    X 为 (N, 28, 28) float64 值域 0~1；y 为 (N,) 整型。
    默认返回官方完整划分（60000 / 10000）。n_train 供冒烟阶段覆盖为 500~1000
    取前若干张；正式训练阶段的 config.N_TRAIN 切片由 Phase 3 训练脚本负责，
    不在此处默认截断（验收要求 X_train 为 (60000,28,28)）。
    """
    log("== load_mnist: 检查/下载 MNIST ==")
    ti, _ = _download_one(_FILES["train_images"], log=log)
    tl, _ = _download_one(_FILES["train_labels"], log=log)
    tei, _ = _download_one(_FILES["test_images"], log=log)
    tel, _ = _download_one(_FILES["test_labels"], log=log)

    X_train = _parse_idx_images(ti)
    y_train = _parse_idx_labels(tl)
    X_test = _parse_idx_images(tei)
    y_test = _parse_idx_labels(tel)

    if n_train is not None:
        X_train, y_train = X_train[:n_train], y_train[:n_train]
    log(f"== load_mnist 完成：X_train{np.shape(X_train)} y_train{np.shape(y_train)} "
        f"X_test{np.shape(X_test)} y_test{np.shape(y_test)} ==")
    return X_train, y_train, X_test, y_test


def to_nchw(X):
    """把 (N, 28, 28) 图像张量补通道维为 (N, 1, 28, 28)（CNN 输入约定 NCHW）。

    统一入口：train / evaluate / app 一律用它，禁止各处自行 reshape。
    已是 4D 则原样返回。
    """
    X = np.asarray(X)
    if X.ndim == 4:
        return X
    if X.ndim == 3:
        return X[:, None, :, :]
    raise ValueError(f"to_nchw 期望 (N,H,W) 或 (N,C,H,W)，实际 {X.shape}")


def add_gaussian_noise(X, sigma=None, rng=None):
    """对图像张量增加高斯噪声并裁剪到 [0,1]。

    sigma 默认取 config.NOISE_SIGMA。
    rng 为 None 时用 np.random.default_rng(config.SEED)（可复现，兼容旧调用）；
    传入外部 rng 时，调用方负责其确定性（训练循环逐 epoch 换噪声以做数据增强，
    整体仍由 config.SEED 派生 → 可复现）。
    """
    if sigma is None:
        sigma = config.NOISE_SIGMA
    if rng is None:
        rng = np.random.default_rng(config.SEED)
    return np.clip(X + sigma * rng.standard_normal(X.shape), 0.0, 1.0).astype(np.float64)


def _otsu_threshold(a):
    """Otsu 阈值：遍历 256 个候选，最大化组间方差 w0*w1*(m0-m1)^2（纯 NumPy）。

    返回 (阈值, 是否可用)。直方图退化（全图同值 / 无有效分割）时 是否可用=False。
    """
    hist, edges = np.histogram(a, bins=_OTSU_BINS, range=(0.0, 1.0))
    total = a.size
    if total == 0:
        return 0.0, False
    centers = 0.5 * (edges[:-1] + edges[1:])
    w0 = np.cumsum(hist).astype(np.float64)          # 背景像素数(累计)
    w1 = total - w0                                  # 前景像素数
    csum = np.cumsum(hist * centers)
    good = (w0 > 0) & (w1 > 0)
    if not np.any(good):
        return 0.0, False
    m0 = np.zeros_like(w0)
    m1 = np.zeros_like(w0)
    m0[good] = csum[good] / w0[good]
    m1[good] = (csum[-1] - csum[good]) / w1[good]
    between = np.where(good, w0 * w1 * (m0 - m1) ** 2, -1.0)
    if between.max() <= 0.0:
        return 0.0, False
    return float(centers[int(np.argmax(between))]), True


def _binarize_otsu(a):
    """A1: Otsu 二值化 -> 输出只含 0.0 / 1.0。退化输入原样返回（不崩）。

    - 直方图退化（近纯色、无有效分割）-> 跳过二值化
    - 二值化后前景占比为 0（或全前景）-> 跳过二值化
    - 阈值极端（>0.9 或 <0.1）-> 仍执行，但打印一条警告
    """
    t, ok = _otsu_threshold(a)
    if not ok:
        return a
    if t > _OTSU_WARN_HI or t < _OTSU_WARN_LO:
        print(f"[to_mnist_format] 警告: Otsu 阈值极端 t={t:.3f}（图像对比度过低?）")
    out = (a > t).astype(np.float64)
    fg = float(out.mean())
    if fg <= 0.0 or fg >= 1.0:
        return a
    return out


def _bbox(mask):
    """由前景掩码求 bounding box，返回 (r0, r1, c0, c1)（半开区间）或 None。"""
    if not mask.any():
        return None
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    return int(rows[0]), int(rows[-1]) + 1, int(cols[0]), int(cols[-1]) + 1


def _crop_and_center(a):
    """A2: 前景裁剪 + 保持长宽比缩放 + 居中到 28×28（纯 NumPy，盒式平均）。

    自适应规则（避免损伤本来就标准的输入）：
      1. 输入已经是 28×28（如标准 MNIST，数据集本身已做 20×20 居中归整）
         -> 跳过裁剪，避免二次归整破坏笔画几何
      2. 前景为空（纯色图）-> 跳过
      3. 用固定阈值 _FG_THRESHOLD 求 bbox；若 bbox 已覆盖画面 >= skip_ratio
         （典型原因：拍照/截图有明显光照梯度，背景也超过阈值）
         -> 改用 Otsu 阈值再求一次 bbox；仍覆盖过大则跳过裁剪
      4. 缩放一律用盒式平均（_resize_box），不用最近邻
    """
    canvas = config.IMG_SIZE
    if a.shape == (canvas, canvas):
        return a
    target = canvas - 2 * _CROP_MARGIN

    box = _bbox(a > _FG_THRESHOLD)
    if box is None:
        return a
    r0, r1, c0, c1 = box
    if (r1 - r0) * (c1 - c0) >= _CROP_SKIP_RATIO * a.size:
        t, ok = _otsu_threshold(a)
        box2 = _bbox(a > t) if ok else None
        if box2 is None:
            return a
        r0, r1, c0, c1 = box2
        if (r1 - r0) * (c1 - c0) >= _CROP_SKIP_RATIO * a.size:
            return a

    crop = a[r0:r1, c0:c1]
    h, w = crop.shape
    scale = target / max(h, w)
    nh = max(1, int(round(h * scale)))
    nw = max(1, int(round(w * scale)))
    small = _resize_box(crop, nh, nw)
    out = np.zeros((canvas, canvas), dtype=np.float64)
    top = (canvas - nh) // 2
    left = (canvas - nw) // 2
    out[top:top + nh, left:left + nw] = small
    return out


def to_mnist_format(img):
    """把任意 numpy 数组规整为 MNIST 风格单通道 (28, 28) float64 0~1。

    不负责文件 IO（文件读取留给 Phase 6 的 Tkinter）。
    处理步骤（Phase 8 改进后的顺序）：
      ①值域>1 则 /255；②RGB→灰度(0.299/0.587/0.114)；
      ③自动反色：均值>0.5 认为白底黑字，取 1-img（MNIST 为黑底白字）；
      ④前景裁剪 + 居中（自适应，已占满则跳过）；⑤盒式平均缩放 28×28；
      ⑥Otsu 二值化（输出只含 0.0/1.0；退化输入跳过）。
    返回 (28, 28) float64，签名与返回类型与旧版一致。
    """
    img = np.asarray(img, dtype=np.float64)
    if img.size == 0:
        raise ValueError("空图像")
    if img.max() > 1.0:
        img = img / 255.0
    if img.ndim == 3:
        if img.shape[-1] in (3, 4):
            img = (0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2])
        else:  # 单通道但有额外维度
            img = img[..., 0]
    if img.ndim != 2:
        raise ValueError(f"无法解释的形状 {img.shape}（需 2D 灰度或 3D RGB）")

    if img.mean() > 0.5:  # 自动反色：白底黑字 -> 黑底白字
        img = 1.0 - img

    img = _crop_and_center(img)          # A2（自适应；已占满/纯色则跳过）

    if img.shape != (config.IMG_SIZE, config.IMG_SIZE):
        img = _resize_box(img, config.IMG_SIZE, config.IMG_SIZE)

    img = _binarize_otsu(img)            # A1（退化输入跳过）
    return img


def _resize_box(img, th, tw):
    """手写盒式平均缩放：源图分 th×tw 网格，每格取均值，禁止 cv2/PIL/scipy。"""
    h, w = img.shape
    out = np.zeros((th, tw), dtype=np.float64)
    for i in range(th):
        y0, y1 = int(i * h / th), int((i + 1) * h / th)
        if y1 <= y0:
            y1 = y0 + 1
        for j in range(tw):
            x0, x1 = int(j * w / tw), int((j + 1) * w / tw)
            if x1 <= x0:
                x1 = x0 + 1
            out[i, j] = img[y0:y1, x0:x1].mean()
    return out


if __name__ == "__main__":
    # 直接运行可快速验证加载
    Xtr, ytr, Xte, yte = load_mnist()
    print("X_train", np.shape(Xtr), Xtr.dtype, "min/max", Xtr.min(), Xtr.max())
    print("X_test ", np.shape(Xte), Xte.dtype)
