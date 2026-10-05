# 实现 Agent 交接提示词：CNN 数字识别项目（NumPy 全手写 + 可视化小窗口）

> 用途：交给负责具体落地实现的 agent。硬约束必须遵守；标注[建议]的可用更优做法替代，但需说明理由。

## 0. 一句话目标
在本仓库从零实现"输入图片 → 灰度化 → 高斯卷积去噪 → CNN 识别数字 0-9"的完整程序：Python + NumPy 全手写，本地 CPU 训练，交付 CLI 训练/评测脚本 + Tkinter 可视化预测窗口。

## 1. 环境事实（已核实，勿重复排查）
- Windows；Python 3.9.13；**numpy 2.0.2、matplotlib 3.9.4 已装**；无 torch/tensorflow（本项目也不许装）
- 机器有 RTX 3060 Laptop 6GB，但本项目规模用 CPU 即可
- 仓库已连 GitHub（origin main，当前空仓库，历史测试内容已全部 Revert）

## 2. 硬约束（违反即返工）
1. **全手写**：高斯核生成、二维卷积、CNN 前向/反向传播、梯度下降一律手写 NumPy。禁止用 scipy/torch/opencv/sklearn 的卷积、滤波、神经网络封装代替（sklearn 仅可用于下载/加载 MNIST）。matplotlib 仅用于画图。
2. **去噪必须真实**：图片真实经历 灰度化 → 加噪（训练时）→ 高斯核卷积去噪 的处理过程，不许装饰性跳过。
3. **目录分离**：`src/` 只放代码实现；一切 agent 思考痕迹（规划/笔记/实验日志/复盘）放 `agent/`；`.gitignore` 排除 `agent/` 与权重文件。
4. **不建五段链路目录**：本项目不建 0-5 目录（用户已明确该纪律只适用于数据/报表类任务，不适用软件开发）。

## 3. 数据与训练
- 数据集：**MNIST**（公有领域开源，28×28 灰度手写数字）。两段式：先小样本冒烟（验证管线正确性 + 梯度检查），再上 1 万张正训；全量 6 万备用。
- 噪声注入：训练/测试时对图像叠加**高斯噪声**（参数放配置，可调）。
- 模型：简单 CNN，2 conv + pool + FC，<100k 参数，CPU 可训。
- 环境：纯 CPU。

## 4. 交付物
1. `src/train.py`（CLI）：训练，保存权重 `outputs/model.npz` + 训练曲线图
2. `src/evaluate.py`（CLI）：测试集准确率 + 混淆矩阵图
3. `src/app.py`（**Tkinter 小窗口**，matplotlib 嵌入 via FigureCanvasTkAgg）：
   - "选择图片"按钮（Tkinter 自带 filedialog 文件选择对话框，零额外依赖；用户已确认不需要拖拽）
   - 用户载入一张（可含高斯噪声的）数字图片后，**逐步/分栏展示三个阶段**：
     ① 信号接收：原图 → 灰度化结果
     ② 信号处理：加噪（可选）/去噪前后对比、所用高斯核热图
     ③ 神经识别：卷积特征图若干 + 最终识别数字与置信度
4. `outputs/`：图表（可提交 git）+ 权重 .npz（gitignore）
5. `README.md` 使用说明；`requirements.txt`（仅 numpy、matplotlib，Tkinter 是标准库）

## 5. 目录骨架
```
神经网络/
├── src/
│   ├── data.py        # MNIST 下载/加载/噪声注入
│   ├── filters.py     # 手写高斯核生成与二维卷积
│   ├── cnn.py         # 手写 CNN（层定义/前向/反向/梯度检查）
│   ├── train.py       # CLI 训练
│   ├── evaluate.py    # CLI 评测
│   └── app.py         # Tkinter 可视化窗口
├── agent/             # agent 工作痕迹（.gitignore 排除）
│   ├── plan/
│   ├── research/
│   └── log/
├── outputs/           # 图表（git 可提交）+ 权重 .npz（gitignore）
├── .gitignore         # 排除 agent/、outputs/*.npz、__pycache__/ 等
├── README.md
└── requirements.txt
```

## 6. 执行步骤
1. 建目录骨架与 `.gitignore`
2. `filters.py` + 冒烟：对小图做真实高斯去噪，肉眼+数值验证
3. `data.py`：MNIST 获取 + 高斯噪声注入
4. `cnn.py`：先做梯度检查（数值梯度 vs 解析梯度，误差 <1e-6 量级），再小样本训练验证收敛
5. `train.py` 全量训练 → `evaluate.py` 出指标（目标 97%+，加噪下可适当放宽）
6. `app.py` 窗口串全流程（三阶段逐步可视化）
7. 全程探索性笔记/实验记录写 `agent/`；最终自测、提交（`agent/` 不进版本库）

## 7. 红线
- 不把 `agent/` 工作提交进版本库（靠 `.gitignore` 强制）
- 不用任何框架封装代替手写（卷积/滤波/反向传播）
- 去噪步骤必须真实执行，不许装饰性跳过
- 跑不动的部分不许假装完成；宣称的指标必须有真实运行输出佐证
