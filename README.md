# 小鼠视网膜 OCT 层边界分割

在小鼠视网膜 OCT B-scan 上自动定位 11 条层边界，并把训练好的模型逐帧用在原始 OCT 视频（AVI）上，得到每条边界的深度图和各层厚度图。

![AVI 上的预测（测试动物，训练时未见过）](results/figures/avi_frame_00100.png)

## 1. 数据

- **来源：** MATLAB Image Labeler 导出的标注（`grdTruthOutput`，每条边界一个 `labelMask`）和对应的原始 TIFF。
- **规模：** 719 张 B-scan，尺寸 1143×782，来自 64 只小鼠、7 个实验组（NaIO3 0/3/7 dpi、tgCR、JHA_CRX29、RPE31_32、RPE34）。
- **数据划分：按动物切分**，同一只动物的所有图像只出现在一个集合里，避免同一只眼睛的相邻切片同时进入训练和测试。

  | | 图像 | 动物 |
  |---|---|---|
  | train | 546 | 35 |
  | val | 88 | 15 |
  | test | 85 | 14 |

- **标注是稀疏的：** 不是每张图都标了全部 11 条边界（例如 GCL 只在 167 张图里出现），一条边界也常常只标了一部分列。没有标注的列不参与训练损失，不会被当成“这里没有这条边界”。
- **重复文件处理：** 不同文件夹里完全相同的图像和标注只保留一份；没有任何有效标注的草稿文件会被跳过。

![训练数据中的标注示例](results/figures/gt_example_1.png)

## 2. 层的规则

### 2.1 11 条边界与顺序

数组坐标中 y 轴向下，玻璃体在图像上方。从上到下，边界顺序固定为：

```
RNFL → GCL → IPL → INL → OPL → ONL → OLM → IS → OS → RPE → CHOROID
```

每个名字表示**该层靠玻璃体一侧的上边界**，例如 “IPL” 指 GCL/IPL 交界，“INL” 指 IPL/INL 交界。

这个顺序不是人工设定的，而是从训练集标注中统计出来的：

- 在数百万个“两条边界在同一列都有标注”的位置上，**顺序反转的次数为 0**；
- 相邻边界的平均间距与小鼠视网膜各层厚度相符，例如 IPL→INL 约 90 px，INL→OPL 约 60 px，OPL→ONL 约 15 px，ONL→OLM 约 110 px。

详细统计见 [`results/data/order_table.csv`](results/data/order_table.csv)。

### 2.2 训练中使用的解剖规则

1. **顺序约束：** 对每一对边界 (a, b)，只要 a 在 b 上方，就要求预测满足 `y_a ≤ y_b`。违反时惩罚 `relu(y_a − y_b)`。
   - 两条边界**重合是允许的**（层可以变得很薄，或在病理区域消失），只有交叉会被惩罚。
   - 共 45 对约束；每对都需要满足：训练集中至少 200 列有共同标注，且反转率 ≤ 1%。
2. **平滑约束：** 对每条边界惩罚相邻列之间的二阶差分（Huber 损失），抑制锯齿和突然跳变。
3. **置信度门控：** 以上两种约束只作用在模型**有把握**的列上，即预测峰附近 ±5 px 内的概率质量 > 0.7。如果某一层在病变或视神经附近真的消失了，模型在那里的预测分布会变平，约束就不会生效，因此不会被强行“画出”一条不存在的线。
4. **CHOROID 特殊处理：** CHOROID 在 82% 的列上标了两条线（脉络膜的上、下界），所以训练目标是双峰分布，并且不参与顺序和平滑约束。推理时取概率最高的两个峰，作为上、下两条曲线。

## 3. 模型与训练

**模型：** FFC-UNet，约 13M 参数。它是一个 5 级残差 U-Net，从第 3 级开始以及在瓶颈层使用 Fast Fourier Convolution，用频域分支获得全图范围的感受野。归一化采用 GroupNorm。输出 11 个通道，每个通道在**每一列上沿 y 方向做 softmax**，得到该边界在这一列的位置分布。

**训练目标：** 每条标注的边界在对应列上转成高斯软标签（σ = 1.5 px）。当一列里连续标注了好几个像素时，高斯会相应加宽。

**损失函数：**

```
L = CE(列方向软标签)  +  λ_order · L_order  +  λ_smooth · L_smooth
λ_order = 0.1,  λ_smooth = 0.01
```

两项约束从第 3 个 epoch 开始启用，并在 3 个 epoch 内线性增加到设定权重。

**解码：** 在每列概率最高的位置附近 ±5 px 内做加权平均（局部 soft-argmax），得到亚像素精度的边界位置。

**训练设置：**

| 项目 | 设置 |
|---|---|
| epochs | 30 |
| batch size | 16（每次前向 4 张，梯度累积 4 次；GroupNorm 下与真实 batch 16 等价） |
| 优化器 | AdamW，lr 3e-4，weight decay 1e-4，梯度裁剪 1.0 |
| 学习率 | 前 5% 步数线性 warmup，之后余弦衰减 |
| 精度 | bf16 混合精度（FFT 部分用 fp32） |
| 数据增强 | 水平翻转；上下平移 ±40 px；gamma、对比度、亮度扰动；高斯噪声。几何增强同步作用在图像和标注上 |
| 模型选择 | 保存验证集 macro W1 最低的 epoch |
| 硬件 | 1× NVIDIA H100，约 16 分钟，显存峰值约 17 GB |

![训练曲线](results/figures/training_curves.png)

## 4. 结果

误差单位为像素，在 1143×782 的原始网格上计算。MAE 只统计 GT 在该列只有一条线的位置；CHOROID 按上、下两条线分别匹配最近的预测峰。

| 边界 | val MAE | val 中位数 | test MAE | test 中位数 |
|---|---|---|---|---|
| RNFL | 2.00 | 1.54 | 2.15 | 1.58 |
| GCL | 3.46 | 1.69 | 3.13 | 1.87 |
| IPL | 4.58 | 2.47 | 4.70 | 2.45 |
| INL | 3.17 | 2.19 | 3.38 | 2.55 |
| OPL | 2.47 | 1.93 | 2.68 | 1.93 |
| ONL | 2.62 | 1.97 | 2.66 | 2.13 |
| OLM | 2.73 | 1.36 | 2.17 | 1.35 |
| IS | 1.97 | 1.27 | 1.66 | 1.19 |
| OS | 2.56 | 1.55 | 4.24 | 1.81 |
| RPE | 1.75 | 1.41 | 5.33 | 1.49 |
| CHOROID（两条线） | 4.00 | 2.01 | 7.00 | 2.17 |
| **平均（不含 CHOROID）** | **2.73** | | **3.21** | |

- 大部分边界的中位误差在 1.2–2.5 px 之间，预测出的边界线连续，基本没有跳变。
- 平均值主要被少数困难图像拉高。例如 test 上 RPE 的中位数是 1.49 px，P90 是 4.5 px，但 P99 达到约 96 px：少数图像上整段边界落到了相邻结构上。
- IPL 是最难的边界：它和 GCL 只相隔约 13 px，且 GCL 标注很少。
- 按实验组的结果见 [`results/metrics/`](results/metrics/)。

![验证集预测示例](results/figures/val_prediction_2.png)

## 5. 在原始 OCT 视频（AVI）上的应用

1. **几何对应：** 训练用的 TIFF 就是 AVI 某一帧的直接裁剪。以 test 动物 MA-N978 为例，TIFF 对应第 59 帧，匹配 NCC = 1.000，缩放 1.0，无需翻转。所以模型可以直接用在 AVI 帧上，不用重采样。
2. **裁剪：** 每帧按 (x=30, y=26, 770×1000) 裁剪，去掉顶部伪影和左侧黑边。
3. **推理：** 每 2 帧推理一次，将所有帧的边界拼起来，得到每条边界的深度图和相邻边界之间的厚度图。图中的纵轴是帧号，未换算成物理距离。

![深度图](results/figures/avi_depth_maps.png)

![厚度图](results/figures/avi_thickness_maps.png)

需要注意：
- 视神经头附近层结构会中断，那里的预测不代表病理。可以用 `07_avi_infer.py --onh` 把该区域排除。
- GCL/IPL 在 AVI 上是断续的；CHOROID 的下边界在信号弱的地方偶尔会在两个峰之间跳动。

## 6. 复现

所有路径集中在 `slurm/env.sh`（Capella，GPU）和 `slurm/barnard_env.sh`（Barnard，CPU）里设置。

```bash
# 1) 从 MATLAB 标注构建数据集，并按动物切分（CPU）
python 02_prepare.py --mat-root "$MAT_BASE" --out "$DATA" --split-mode animal --workers 8
python 03_review_overlays.py --data-dir "$DATA" --out ~/octb_review --n 20      # 检查标注叠加图

# 2) 训练（GPU，Capella）
sbatch --export=ALL,EXP=E1,MICRO=4 slurm/train.sh

# 3) 评估 + 误差分析（CPU）
python 05_evaluate.py --ckpt "$RUNS/E1/best.pt" --data-dir "$DATA" --split val --out "$RUNS/E1/eval_val"
python 08_analyze.py  --data-dir "$DATA" --models E1="$RUNS/E1/best.pt" --split test --out "$RUNS/analysis_test"

# 4) AVI
python list_avi.py --split test                                   # 列出 test 动物的 AVI 和对应 TIFF
python 06_avi_match.py --avi "$AVI" --tiff "$TIF" --data-dir "$DATA" --out match.json
python 07_avi_infer.py --avi "$AVI" --ckpt "$RUNS/E1/best.pt" --data-dir "$DATA" --match match.json --out avi_out --step 2

# 5) 把结果收集到 results/
python make_report_assets.py --avi-name <AVI 文件名，不含 .avi>
```

训练好的权重在 `results/model/E1_weights.pt`（fp16，仅权重）。它可以直接作为 `--ckpt` 传给 `05_evaluate.py`、`07_avi_infer.py`；如果数据集不是同一次构建的，会出现 manifest 不一致的警告，可以忽略。

依赖：Python 3.11、PyTorch ≥ 2.1、numpy、scipy、matplotlib、Pillow、tifffile、opencv-python-headless。

## 7. 代码结构

```
02_prepare.py          MATLAB 标注 → 数据集；按动物切分；统计边界顺序
03_review_overlays.py  标注叠加图
04_train.py            训练（CE + 顺序约束 + 平滑约束）
05_evaluate.py         评估：W1、MAE、交叉率、跳变率，逐动物结果
06_avi_match.py        在 AVI 中找到 TIFF 对应的帧、缩放和翻转
07_avi_infer.py        AVI 逐帧推理，生成深度图和厚度图
08_analyze.py          误差分析：CHOROID 双线评估、按实验组统计、最差图像
list_avi.py            列出 AVI 及其所属 split
make_report_assets.py  收集结果到 results/
octb/                  数据读取、软标签、模型、损失、指标
slurm/                 作业脚本
tests/                 合成数据端到端测试（bash tests/smoke_test.sh）
results/               图、指标表、训练记录、模型权重
```
