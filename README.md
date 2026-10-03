# 手写数字识别（全手写 NumPy 实现）

## 1. 项目简介

输入一张图片 → 预处理（自动反色、裁剪居中、缩放 28×28、Otsu 二值化）→ 高斯卷积去噪 → CNN 识别数字 0-9。

**全部核心算法手写**：二维卷积、卷积反向传播、最大池化、Softmax 与交叉熵、SGD 训练循环，
一律只用 **NumPy** 实现，**不使用任何深度学习框架**（无 torch / tensorflow / sklearn / scipy / cv2 / PIL）。
训练在**本地 CPU** 上完成，模型参数量约 **9,098**（远小于 100k）。

网络结构（NCHW）：

```
输入 (N,1,28,28)
 → Conv(1→8, 3×3, zero-pad same) → ReLU → MaxPool(2×2)   → (N,8,14,14)
 → Conv(8→16, 3×3, zero-pad same) → ReLU → MaxPool(2×2)  → (N,16,7,7)
 → Flatten                                                → (N,784)
 → FC(784→10) → Softmax                                   → (N,10)
```

## 2. 环境要求

- Python **3.9+**（本项目在 3.9.13 上开发验证）
- 依赖仅两项：**numpy**、**matplotlib**（Tkinter 属标准库，无需安装）

```bash
pip install -r requirements.txt
```

`requirements.txt` 内容（仅两行）：

```
numpy==2.0.2
matplotlib==3.9.4
```

## 3. 目录结构

```
src/                        9 个扁平模块（无子包）
  config.py                 全项目唯一参数来源（尺寸/超参/路径/随机种子）
  filters.py                手写高斯核、二维卷积（朴素+向量化+批量）、卷积反向
  data.py                   MNIST 手写 IDX 解析、通道维、噪声注入、图像预处理（反色/裁剪居中/缩放/Otsu）
  layers.py                 Conv / ReLU / MaxPool / Flatten / FC / Softmax（前向+反向）
  cnn.py                    组网 + 数值稳定交叉熵 + 反向传播 + 梯度检查
  train.py                  CLI 训练（真实去噪管线 + SGD + 存权重 + 训练曲线）
  evaluate.py               CLI 评测（从权重复现准确率 + 混淆矩阵）
  viz.py                    绘图工具（曲线/混淆矩阵/去噪对比/对照柱状图）
  app.py                    Tkinter 五步可视化窗口 + MNIST 图片库（含 --selftest 自检）
data/mnist/                 MNIST 原始数据（4 个 .gz，已提交进版本库）
outputs/                    训练曲线、混淆矩阵、去噪对比、对照图；model*.npz（不进版本库）
```

## 4. 快速开始

```bash
python src/train.py                              # 完整训练（10000 张 / 10 epochs）
python src/train.py --n-train 1000 --epochs 2 --tag smoke   # 冒烟（约 36 秒）
python src/evaluate.py                           # 用已有权重复现准确率
python src/app.py                                # 启动可视化窗口
```

> ⚠ **产物覆盖警告**：`train.py` 总是把结果写到固定名字
> `outputs/model{_tag}.npz`、`train_curve{_tag}.png`、`confusion_matrix{_tag}.png`。
> **不带 `--tag` 的运行会覆盖正式权重 `outputs/model.npz`。**
> 所以做冒烟/试验时**务必加 `--tag`**（如 `--tag smoke` → 写 `model_smoke.npz`），
> 否则会把训练好的权重覆盖成小模型，且因为 `*.npz` 不进版本库而**无法用 git 回滚**。

也可以**直接双击根目录的 `启动窗口.bat`**（无需命令行，用 `pythonw` 启动，无黑框）。

其它可用开关：

```bash
python src/train.py --no-noise --no-denoise      # 组 A：干净输入（上界）
python src/train.py --no-denoise                 # 组 C：加噪不去噪（消融）
python src/evaluate.py --model outputs/model_A.npz --no-noise --no-denoise
python src/app.py --selftest <图片> --out-prefix outputs/app_selftest   # 分步面板图
```

## 5. 三组对照实验的结果（本项目最重要的结论）

统一配置：`n_train=20000, epochs=15, lr=0.03, batch=64, seed=42`，测试集 10000 张。

| 组 | 训练 / 测试输入 | 测试准确率 |
|---|---|---|
| A 基准 | 干净 → 干净 | **97.89%** |
| B 主线 | 加噪 + 高斯去噪 | **96.57%** |
| C 消融 | 加噪（不去噪） | **97.21%** |

**结论（如实呈现，未美化）：**

- 去噪让**像素级 MSE** 从 **0.0470** 降到 **0.0321**（**下降 32%**），即"看起来更接近干净图"；
- 但识别率**反而下降 0.64 个百分点**（C 97.21% > B 96.57%）；
- 原因：噪声残余是**零均值随机项**（后续卷积可部分抵消），而高斯模糊带来的偏差是
  **系统性的信息丢失**（细笔画高频被抹掉，不可恢复）；
- **教训：不能用 MSE / std 下降来论证"去噪对下游任务有益"。** 评价指标必须落在下游任务上。

> 补充：这也是 Phase 1 用**合成平滑环图**演示去噪时被掩盖的代价——合成环本身平滑、
> 没有细笔画可毁，所以当时 MSE 指标好看。换成真实手写数字的细笔画后才暴露。
> 去噪核为 5×5、σ=1.2，其平滑半径与 MNIST 笔画宽度**同量级**，是代价偏大的直接原因。

> 复现方式：`model_A.npz` / `model.npz` / `model_C.npz` 三个权重对应上表三组，
> 分别用 `python src/evaluate.py --model outputs/model_A.npz --no-noise --no-denoise`
> （A）、`python src/evaluate.py`（B）、
> `python src/evaluate.py --model outputs/model_C.npz --no-denoise`（C）复现。
> 对应的 `train_curve_A/C.png`、`confusion_matrix_A/C.png` 见 `outputs/`。
> 如需从零复训，用第 5 节配置：`--n-train 20000 --epochs 15 --lr 0.03`（单组约 15 分钟）。

## 6. 使用边界（实测，最重要的一节）

测试协议：把 MNIST 测试集前 500 张，按下列形态合成到 140×140 画布（≈5 倍放大，模拟截图/拍照），
再走完整管线（`to_mnist_format` → 加噪 → 去噪 → `model.npz` 前向）。
**测量脚本已留档**，可随时重跑复核：

```bash
python agent/experiments/measure_boundary.py 500
```

| 输入形态 | 改进前 | **改进后（当前）** |
|---|---|---|
| 数字占满画面（MNIST 风格） | 96.4% | **96.4%**（无损） |
| 数字只占画面 1/3、居中（白纸黑字） | 9.2% | **88.6%** |
| 数字偏在角落（白纸黑字，占 1/3） | 6.4% | **88.6%** |
| 白纸手写拍照，数字占 70%、光照不均 | 68.4% | **63.4%** |
| 白纸手写拍照，数字占 30% | 8.4% | **11.0%** |
| 大图满幅 280×280（反直觉的边界） | — | **87.8%** |

**结论：**

- **"数字太小 / 偏在角落"已被解决**：新的预处理会**自动裁剪出数字并居中**，
  把 9.2% / 6.4% 提升到 **88.6%**（两种形态结果相同，正因为裁剪后都归到了同一位置）；
- **对标准输入零损伤**：占满画面仍是 **96.4%**，与改进前完全一致；
- **光照不均的拍照图仍偏弱**：预处理靠阈值找数字边界，背景亮度不匀时容易被带偏。

**实用建议：**

- **最优**：用画图软件写一个数字后截屏，让数字**基本填满画面** → 约 **96%**；
- **次优（最省事）**：白纸黑字随便写，**不用特意裁剪**，数字只占画面 1/3 也有 **88.6%**；
- **拍照**：尽量让光线均匀、让数字占画面 2/3 以上 → 约 **63%**；
  远距离拍整张纸（数字占 30%）仍会掉到 **11%**，此时请先手动裁剪再识别。

> 注：光照不均一组的绝对数值对模拟细节（光照梯度方向、模糊强度）较敏感，不同脚本会给出
> 60%~79% 不等的结果；本表以**留档脚本**的数字为准（`agent/experiments/measure_boundary.py`）。

## 7. 窗口使用说明

命令行 `python src/app.py`（或双击 `启动窗口.bat`）后，界面按**五步**推进，
每点一次「下一步」才执行该步计算（不是一次算完再翻页）：

| 步骤 | 内容 |
|---|---|
| ① 收到图片 | 左：原图缩略；右：预处理结果 = 灰度化 + 自动反色 + **裁剪居中** + 缩放 28×28 + Otsu 二值化（标注 `raw_gray_mean` / `auto_inverted` / `gray_mean`） |
| ② 加噪 | 左：28×28 干净图；右：加噪图（标注 σ、加噪后 std 与 MSE），可勾选关闭 |
| ③ 去噪 | 加噪图 / 去噪图 / 所用的 5×5 高斯核热图（标注核和=1） |
| ④ CNN 逐层 | input → conv1(2×4) → pool1(2×4) → conv2(4×4) → pool2(4×4) → flatten → softmax 全部中间激活；每张小图标注该通道 std，并给出该层 min/max |
| ⑤ 识别结果 | 大字号预测数字 + 置信度；下方 0-9 概率柱状图（最高柱高亮、柱顶标 4 位小数、标题含 Top-2，置信度 <0.6 时提示低置信度） |

**三个图片入口**（三条路径都汇入同一个 `_load_rgb()`，走完全相同的预处理与五步管线）：

- **选择图片**：打开文件对话框，支持 PNG / GIF / PPM / PGM（不支持的格式会明确提示换格式）
- **从剪贴板粘贴（Ctrl+V）**：配合 `Win+Shift+S` 截屏，回到窗口按 `Ctrl+V` 或点按钮即可载入
- **从数据集选择**：打开 **MNIST 图片库**窗口 —— 顶部按标签检索（「全部」+ 0~9），
  中间是 6×6 缩略图网格（每张标注测试集索引 `#N` 与真实标签 `Lk`），「换一批」换一组；
  **点任一缩略图即载入**主窗口，状态栏显示如 `MNIST test #127 (label 5)`

命令行等价用法（无窗口自检，用于脚本化验证）：

```bash
python src/app.py --dataset 127 --label 5 --out-prefix agent/experiments/selftest_ds
# --dataset N：MNIST 测试集第 N 张；加 --label K 表示"标签 K 下的第 N 张"
```

## 8. 已知限制

- 只识别**单个**数字：不支持多位数、整行文字；
- 不适用于印刷体、复杂背景、带边框/水印的图；
- 极细笔画、低对比度、强光照不均的拍照图效果差（见第 6 节，拍照 70% 约 **63%**）；
- **大图里数字几乎占满整幅时反而略差**（实测 280×280 满幅 **87.8%**，低于 28×28 原生图的 96.4%）：
  预处理会对大图做裁剪居中，而对已经归整过的图再裁一次，会轻微改变笔画几何；
- 预处理能自动**反色 / 裁剪居中 / 缩放 / 二值化**，但**不做透视校正、不做去水印、不做多数字分割**；
- 训练只用了 MNIST 训练集的前 20000 张、15 个 epoch（CPU 可承受的范围），未追求 SOTA；
- `data/` 中的 MNIST 为官方 IDX 格式，解析全部手写实现。

## 9. 实现约束与合规

- **手写**：二维卷积、卷积反向传播、最大池化及其反向、Softmax+交叉熵、SGD —— 无框架参与；
- **未使用**：torch / tensorflow / sklearn / scipy / cv2 / PIL / `np.fft`（依赖仅 numpy + matplotlib）；
- **边界策略统一**：全部卷积为 zero-pad / same，前向与反向同源（`filters.py`）；
- **参数集中**：所有可调参数来自 `src/config.py`，不在别处散落魔法数字；
- **不进版本库**：`agent/`（工作痕迹）、`third_party/`（参考代码）、`*.npz`（权重）、
  `__pycache__/`、`outputs/log_*.txt`（训练日志）。

**数字可复核**：第 6 节的"使用边界"表由 `agent/experiments/measure_boundary.py` 实测生成
（该脚本位于 `agent/` 下、不进版本库，但随项目保留在本地，可随时重跑复核）。
