# MEL-StarNet 项目说明

## 项目简介

本项目实现了你设计的 `MEL-StarNet` 网络，并配套整理出一套可直接用于论文实验的代码体系。当前工程已经覆盖：

- `MEL-StarNet` 主模型实现
- 原始 `StarNet_s4` 基线与多组消融/替换实验模型
- 训练、跑通验证、参数量/FLOPs 统计、论文表格生成脚本
- 按论文表格名称整理好的模型文件夹，方便直接对应实验章节

这套代码更偏向“论文实验工程”，目标不是单一训练脚本，而是把模型实现、消融实验、对比实验、结果整理全部打通。

## 项目亮点

- 主模型 `MEL-StarNet` 已独立实现，结构包含 `DS-CBS`、`LED-CG`、`DCAFE`、`Inv-Leaf`
- 所有实验模型已经按论文表名分组整理
- 提供统一 `model_zoo` 接口，便于按模型名创建网络
- 提供训练脚本、跑通脚本、FLOPs/参数量统计脚本、论文表格生成脚本
- `train.py` 支持直接输入训练集和验证集路径，并带 batch 级进度显示

## 目录结构

```text
.
├─ 公共核心
│  ├─ mel_starnet.py
│  ├─ mel_starnet_variants.py
│  ├─ model_zoo.py
│  └─ starnet.py
├─ 公共脚本
│  ├─ train.py
│  ├─ test_mel_starnet.py
│  ├─ profile_models.py
│  └─ generate_paper_tables.py
├─ 表4_2_实验结果对比表
├─ 表4_3_总体模块累加式消融对比表
├─ 表4_4_总体模块去除式消融对比表
├─ 表4_5_DCAFE与经典注意力模块替换对比表
├─ 表4_6_DS_CBS小消融对比表
├─ 表4_7_DCAFE内部小消融对比表
├─ 表4_8_Inv_Leaf小消融对比表
├─ paper_tables
├─ paper_tables_reorganized
└─ runs
```

## 核心文件说明

### 公共核心

- `公共核心/mel_starnet.py`
  - `MEL-StarNet` 主模型实现文件
  - 包含基础模块与主干网络结构

- `公共核心/mel_starnet_variants.py`
  - 所有变体模型的统一实现文件
  - 包含主模型、累加式消融、去除式消融、注意力替换、小消融等版本

- `公共核心/model_zoo.py`
  - 统一模型注册表
  - 可通过模型名直接获取对应模型

- `公共核心/starnet.py`
  - 原始 StarNet 参考实现
  - 主要用于对照和结构参考

### 公共脚本

- `公共脚本/train.py`
  - 通用训练脚本
  - 使用 `ImageFolder` 数据集组织方式
  - 支持运行时输入训练集/验证集路径
  - 支持 batch 进度显示和每轮指标汇总输出

- `公共脚本/test_mel_starnet.py`
  - 快速验证模型能否正常前向运行
  - 包含中间层 shape 检查、模型结构概览、参数量/FLOPs 统计

- `公共脚本/profile_models.py`
  - 通用模型统计脚本
  - 用于评估模型的参数量、MACs、FLOPs

- `公共脚本/generate_paper_tables.py`
  - 自动生成论文表格
  - 输出 Markdown、CSV、指标模板 JSON

## 各表对应模型文件总表

| 所属表格 | 文件夹 | 主要用途 | 关键模型文件 |
| --- | --- | --- | --- |
| 表 4.2 实验结果对比表 | `表4_2_实验结果对比表` | 主对比实验与基线入口 | `benchmark_models.py`、`baseline_starnet.py` |
| 表 4.3 总体模块累加式消融对比表 | `表4_3_总体模块累加式消融对比表` | 模块逐步叠加实验 | `ds_cbs_starnet.py`、`ds_led_cg_starnet.py`、`ds_led_cg_dcafe_starnet.py` |
| 表 4.4 总体模块去除式消融对比表 | `表4_4_总体模块去除式消融对比表` | 去除模块后的性能对比 | `wo_ds_cbs_mel_starnet.py`、`wo_led_cg_mel_starnet.py`、`wo_dcafe_mel_starnet.py`、`wo_inv_leaf_mel_starnet.py` |
| 表 4.5 DCAFE 与经典注意力模块替换对比表 | `表4_5_DCAFE与经典注意力模块替换对比表` | SE/ECA/CBAM/CA/scSE 对比 | `se_mel_starnet.py`、`eca_mel_starnet.py`、`cbam_mel_starnet.py`、`ca_mel_starnet.py`、`scse_mel_starnet.py` |
| 表 4.6 DS-CBS 小消融对比表 | `表4_6_DS_CBS小消融对比表` | `CBS` 与 `DS-CBS` 对比 | `original_cbs_mel_starnet.py` |
| 表 4.7 DCAFE 内部小消融对比表 | `表4_7_DCAFE内部小消融对比表` | `Avg-CA`、`Max-CA`、双分支 DCAFE 对比 | `avg_ca_mel_starnet.py`、`max_ca_mel_starnet.py` |
| 表 4.8 Inv-Leaf 小消融对比表 | `表4_8_Inv_Leaf小消融对比表` | `Involution only` 与 `Residual Involution` 对比 | `involution_only_mel_starnet.py` |

## 已注册的主要模型名

这些名字可以直接用于 `model_zoo.get_model(...)`：

- `ResNet50`
- `DenseNet121`
- `EfficientNet-B0`
- `MobileNetV2-1.4`
- `GhostNetV3 1.3`
- `FasterNet-T1`
- `MobileOne-S2`
- `StarNet_s4`
- `+ DS-CBS`
- `+ DS-CBS + LED-CG`
- `+ DS-CBS + LED-CG + DCAFE`
- `Ours`
- `w/o DS-CBS`
- `w/o LED-CG`
- `w/o DCAFE`
- `w/o Inv-Leaf`
- `Original CBS`
- `SE`
- `ECA`
- `CBAM`
- `CA`
- `scSE`
- `Avg-CA only`
- `Max-CA only`
- `Avg-CA + Max-CA (DCAFE)`
- `Involution only`
- `Residual Involution (Inv-Leaf)`

## 环境依赖

推荐环境：

- Python `3.8`
- PyTorch `2.4.1`
- torchvision `0.19.1`

安装依赖：

```bash
pip install -r requirement.txt
```

当前 `requirement.txt` 中已经包含：

- `torch`
- `torchvision`
- `Pillow`
- `tqdm`
- `thop`
- `torchsummary`
- `timm`

## 数据组织方式

训练脚本默认使用 `ImageFolder` 目录格式：

```text
dataset/
├─ train/
│  ├─ class_1/
│  ├─ class_2/
│  └─ ...
└─ val/
   ├─ class_1/
   ├─ class_2/
   └─ ...
```

要求：

- `train` 和 `val` 中类别文件夹名称一致
- 每个类别文件夹内放对应类别图片

## 常用命令

### 1. 训练主模型

直接运行并手动输入路径：

```bash
python 公共脚本/train.py
```

命令行直接传路径：

```bash
python 公共脚本/train.py --train_dir 你的训练集路径 --val_dir 你的验证集路径
```

训练时终端会显示：

- 每个 batch 的训练/验证进度
- 每轮结束后的 `Train Loss`、`Train Acc`、`Val Loss`、`Val Acc`
- `Precision`、`Recall`、`F1`
- 最优模型更新信息

### 2. 快速验证模型是否跑通

```bash
python 公共脚本/test_mel_starnet.py --num_classes 10 --device cpu
```

### 3. 评估参数量和 FLOPs

统计指定模型：

```bash
python 公共脚本/profile_models.py --models Ours "w/o DCAFE" "SE" --num_classes 10 --device cpu
```

统计全部已注册模型：

```bash
python 公共脚本/profile_models.py --all_zoo --num_classes 10 --device cpu
```

### 4. 自动生成论文表格

先生成参数/FLOPs 表格和指标模板：

```bash
python 公共脚本/generate_paper_tables.py --num_classes 10 --device cpu --output_dir paper_tables --write_metrics_template
```

如果你已经获得各模型的 `Accuracy / Precision / Recall / F1`，可以把结果写入 `metrics_template.json` 后再生成最终表格：

```bash
python 公共脚本/generate_paper_tables.py --num_classes 10 --device cpu --output_dir paper_tables --metrics_file paper_tables/metrics_template.json
```

## 模型导入示例

### 1. 直接导入主模型

```python
from 公共核心.mel_starnet import MELStarNet

model = MELStarNet(num_classes=10)
```

### 2. 通过统一注册表创建模型

```python
from 公共核心.model_zoo import get_model

model = get_model("Ours", num_classes=10)
ablation_model = get_model("w/o DCAFE", num_classes=10)
attention_model = get_model("ECA", num_classes=10)
```

### 3. 按单个实验文件导入

```python
from 表4_5_DCAFE与经典注意力模块替换对比表.eca_mel_starnet import starnet_s4_eca

model = starnet_s4_eca(num_classes=10)
```

```python
from 表4_3_总体模块累加式消融对比表.ds_led_cg_starnet import starnet_s4_ds_led_cg

model = starnet_s4_ds_led_cg(num_classes=10)
```

## 复现实验推荐流程

你可以按照下面这条路径把整个实验流程跑通：

1. 安装环境依赖
   - `pip install -r requirement.txt`

2. 准备数据集
   - 按 `ImageFolder` 格式整理 `train/` 和 `val/`

3. 训练模型
   - 先训练 `Ours`
   - 再根据表格需要训练各个消融和替换模型

4. 验证模型能否正常运行
   - 使用 `test_mel_starnet.py` 做前向与结构检查

5. 统计模型复杂度
   - 使用 `profile_models.py` 统计参数量和 FLOPs

6. 汇总论文结果表
   - 使用 `generate_paper_tables.py` 生成 Markdown/CSV 表格

用更直观的方式表示就是：

```text
安装依赖
  ↓
整理 train/val 数据集
  ↓
训练模型
  ↓
检查模型能否正常跑通
  ↓
统计 Params / FLOPs
  ↓
整理 Accuracy / Precision / Recall / F1
  ↓
生成论文表格
```

## 输出结果说明

### 训练输出

默认保存在：

```text
runs/mel_starnet/
```

常见文件：

- `best.pt`
- `last.pt`
- `class_to_idx.json`

### 论文表格输出

默认保存在：

```text
paper_tables/
```

常见文件：

- `paper_tables.md`
- `table_4_2.csv` 到 `table_4_8.csv`
- `metrics_template.json`

## 使用时的几个说明

- 当前 `train.py` 默认训练的是 `MEL-StarNet` 主模型，也就是 `Ours`
- 部分基线模型依赖 `timm`，例如 `GhostNetV3`、`FasterNet`、`MobileOne`
- 如果 Windows 终端出现中文乱码，通常只是终端编码显示问题，不影响脚本本身运行
- 本项目已经按“论文表格名”整理好文件夹，更适合论文实验管理，而不是普通单模型项目结构

## 建议

如果你下一步还要继续完善项目，比较值得补充的内容有：

- 增加 `predict.py`，用于加载 `best.pt` 做单张图片预测
- 增加 `--model_name` 参数，让 `train.py` 可以直接训练任意消融模型
- 增加批量训练/批量测试脚本，自动回填论文表格指标
