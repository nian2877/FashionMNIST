# FashionMNIST Learning Project

这个仓库按“从入门到项目”的节奏搭建，目前已经覆盖：

- 第 1 天：理解 PyTorch 最小训练闭环
- 第 2 天：加载真实 FashionMNIST 数据并跑通第一版图像分类
- 第 3 天：从 MLP 升级到 CNN，并完成基础模型对比
- 第 4 天：围绕 CNN 做实验分析，包括学习率、batch size、dropout 与错误样本分析
- 第 5 天：做迁移学习，优先使用 `torchvision/timm` 官方预训练骨干，也保留前几天训练好的 CNN 作为后备方案

## 当前文件说明

- `main.py`：项目统一入口
- `dataset.py`：下载、校验并解析 FashionMNIST 原始数据文件
- `model.py`：包含 MLP 基线与 CNN 模型
- `train.py`：第 4 天实验脚本，负责跑对比实验并保存分析结果
- `transfer_dataset.py`：第 5 天真实图片文件夹数据集读取
- `transfer_learning.py`：第 5 天迁移学习脚本
- `DAY2_TUTORIAL.md`：第 2 天教学文档
- `DAY3_TUTORIAL.md`：第 3 天教学文档
- `DAY4_TUTORIAL.md`：第 4 天详细学习文档
- `DAY5_TUTORIAL.md`：第 5 天详细学习文档

## 第 5 天学习目标

今天你会开始接触真正的迁移学习流程：

- 加载一个官方预训练视觉骨干，例如 `ResNet18`
- 在新的图片分类任务上先冻结骨干，再训练新分类头
- 再解冻全模型做微调
- 观察“从头训练”和“迁移学习”在训练效率上的区别

## 如何运行

先确认 `torch`、`matplotlib` 和 `PIL` 可用：

```bash
python -c "import torch, matplotlib; from PIL import Image; print(torch.__version__, matplotlib.__version__)"
```

第 5 天的默认入口是：

```bash
python main.py
```

如果你还想跑第 4 天实验，可以继续使用：

```bash
python train.py
```

也可以直接运行第 5 天脚本：

```bash
python transfer_learning.py
```

## 第 5 天数据集结构

第 5 天要求你准备一个小型真实图片文件夹数据集，结构类似：

```text
data/transfer_real/
  train/
    class_a/
      001.jpg
      002.jpg
    class_b/
      001.jpg
  val/
    class_a/
      101.jpg
    class_b/
      101.jpg
```

如果没有 `val/`，也可以用 `test/` 作为验证集。

当前代码支持三种骨干来源：

- `torchvision_resnet18`
- `timm_resnet18`
- `custom_cnn`

默认使用 `torchvision_resnet18`。如果第一次运行下载预训练权重失败，可以改成不使用预训练权重，或者切到 `custom_cnn` 后备方案。

## 第 5 天输出内容

训练完成后，结果会保存到：

```text
results/day5_transfer/
```

其中会包含：

- 两阶段迁移学习 history JSON
- 训练曲线图
- 错误样本图
- 迁移学习后的模型权重
- 总结文件 `day5_summary.json`

## 为什么第 5 天很重要

很多真实任务里，你的数据不够大，不适合从零训练一个很强的模型。
迁移学习的核心价值就是：

- 复用旧任务已经学到的特征
- 少量数据也能较快起步
- 先训练头部，再微调整网，是非常常见的工程流程

第 5 天就是把项目从“会自己训一个模型”推进到“会迁移已有模型解决新任务”。
