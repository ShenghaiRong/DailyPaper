---
title: "Image Generators are Generalist Vision Learners"
method_name: "Vision Banana"
authors: [Valentin Gabeur, Shangbang Long, Songyou Peng, Paul Voigtlaender, Shuyang Sun, Yanan Bao, Karen Truong, Zhicheng Wang, Wenlei Zhou, Jonathan T. Barron, Kyle Genova, Nithish Kannen, Sherry Ben, Yandong Li, Mandy Guo, Suhas Yogin, Yiming Gu, Huizhong Chen, Oliver Wang, Saining Xie, Howard Zhou, Kaiming He, Thomas Funkhouser, Jean-Baptiste Alayrac, Radu Soricut]
year: 2026
venue: arXiv
tags: [generative-vision-pretraining, visual-understanding, image-generation, segmentation, metric-depth, surface-normal, foundation-model]
zotero_collection: _inbox
image_source: local
arxiv_html: https://arxiv.org/html/2604.20329
created: 2026-04-26
---

# 论文笔记：Image Generators are Generalist Vision Learners

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | Google / Google DeepMind |
| 日期 | April 2026 |
| 项目主页 | https://vision-banana.github.io/ |
| 对比基线 | [[Segment Anything Model 3]], [[Depth Anything V3]], [[Lotus-2]], [[Nano Banana Pro]] |
| 链接 | [arXiv](https://arxiv.org/abs/2604.20329) / [PDF](https://arxiv.org/pdf/2604.20329) |

---

## 一句话总结

> Vision Banana 证明强图像生成器可通过轻量指令微调转化为通用视觉理解模型。

---

## 核心贡献

1. **生成器作为视觉基础模型**: 论文把 [[Image Generation|图像生成]] 预训练类比为 LLM 预训练，主张强生成器内部已经学到可迁移的 [[Visual Understanding|视觉理解]] 表征。
2. **RGB 统一输出接口**: 通过 [[RGB Output Parameterization|RGB 输出参数化]]，把分割、深度、法向量等任务都改写成可解码的图像生成任务。
3. **轻量 instruction tuning 得到通用模型**: [[Vision Banana]] 仅在 [[Nano Banana Pro]] 原训练混合中加入低比例视觉任务数据，就在 2D/3D 多个 benchmark 上达到或接近 SOTA，同时基本保留生成能力。

---

## 问题背景

### 要解决的问题

传统 [[Computer Vision|计算机视觉]] 通常为每个任务设计专门架构、损失函数和输出头，例如 [[Semantic Segmentation|语义分割]] 模型输出类别图，[[Monocular Metric Depth Estimation|单目米制深度估计]] 模型输出连续深度，[[Surface Normal Estimation|表面法向估计]] 模型输出三维向量。本文想回答：强 [[Generative Vision Pretraining|生成式视觉预训练]] 是否已经学到了足够通用的视觉表示，只需要对齐输出格式即可完成理解任务？

### 现有方法的局限

先前直接利用图像/视频生成器做视觉任务时，常能生成“看起来像”深度图或分割图的结果，但格式不稳定，难以可靠解码并在标准 benchmark 上量化。另一类方法会在生成模型上增加专用模块并全量微调，虽然能在单任务上变强，但牺牲了 [[Generalist Model|通用模型]] 的跨任务和生成能力。

### 本文的动机

LLM 的经验说明，生成式预训练模型可以通过 [[Instruction Tuning|指令微调]] 学会遵循任务格式。本文把同样思想迁移到视觉：保留生成图像这个接口，只让模型生成有严格颜色编码的任务结果图。

---

## 方法详解

### 模型架构

[[Vision Banana]] 采用 **生成式通用视觉模型** 架构：

- **输入**: 原始图像 + 自然语言任务指令，例如“用给定 color map 生成语义分割图”。
- **Backbone**: [[Nano Banana Pro]] 图像生成模型。
- **核心模块**: [[Instruction Tuning]] 对齐输出格式；[[RGB Output Parameterization]] 把不同任务输出编码为图像。
- **输出**: 可解码的 RGB 图像，包括 mask、depth visualization、surface normal map 或普通生成图像。
- **总参数**: 论文未披露。

### 核心模块

#### 模块1: 视觉任务的 RGB 输出参数化

**设计动机**: 用 [[Image Generation|图像生成]] 作为统一接口，避免为每个任务增加专门输出头。

**具体实现**:
- [[Semantic Segmentation|语义分割]]: prompt 指定类别到 RGB/hex 颜色的映射，模型输出彩色像素标签图。
- [[Instance Segmentation|实例分割]]: 每次指定一个类别，模型给不同实例分配不同颜色，再通过颜色聚类解码实例 mask。
- [[Referring Expression Segmentation|指代表达分割]]: 用自然语言描述目标对象，模型输出对应区域的 mask 图。
- [[Monocular Metric Depth Estimation|单目米制深度估计]]: 先把深度值通过可逆变换映射到 RGB，再从生成图像反解回米制深度。
- [[Surface Normal Estimation|表面法向估计]]: 将相机坐标系下的法向量分量直接编码到 RGB 通道。

#### 模块2: 低比例视觉数据混合微调

**设计动机**: 只教模型“如何按格式输出视觉任务结果”，而不是重新学习视觉世界。

**具体实现**:
- 将视觉任务数据以很低比例混入 [[Nano Banana Pro]] 原始生成训练混合。
- 2D 数据来自 web-crawled 图像的内部模型标注。
- 3D 数据来自渲染引擎合成数据。
- 论文声明不使用评测 benchmark 的训练集，以保证 [[Zero-Shot Transfer|zero-shot transfer]] 设定。

---

## 关键公式

### 公式1: [[Power Transform|深度弯曲变换]]

$$
f(d, \lambda, c) = 1 - \left(1 - \frac{d}{\lambda c}\right)^{\lambda + 1}
$$

**含义**: 把无界米制深度 $d \in [0, \infty)$ 压缩到 $[0, 1)$，让近处深度获得更高分辨率，再沿 RGB cube 边进行颜色插值。

**符号说明**:
- $d$: 每个像素的米制深度值。
- $\lambda$: power transform 的形状参数，论文约束 $\lambda < -1$，实验中设为 $-3$。
- $c$: 尺度参数，实验中设为 $10/3$。
- $f(d, \lambda, c)$: 归一化后的弯曲深度。

### 公式2: [[RGB Output Parameterization|深度到 RGB 的双射]]

$$
d \xrightarrow{\ f(d,\lambda,c)\ } u \in [0,1) \xrightarrow{\text{RGB cube edge interpolation}} (r,g,b) \in [0,1]^3
$$

**含义**: 论文没有给出完整分段插值公式，但明确说明使用沿 RGB cube 边的 piecewise-linear function，形成从米制深度到 RGB 颜色的可逆映射。

**符号说明**:
- $u$: 弯曲后的归一化深度。
- $(r,g,b)$: 模型生成的深度可视化颜色。
- inverse mapping: 通过投影到最近线段并反向插值，将 RGB 解码回深度。

### 公式3: [[Surface Normal Estimation|表面法向 RGB 编码]]

$$
\mathbf{n} = (x, y, z), \quad x,y,z \in [-1,1]
$$

**含义**: surface normal 本身与 RGB 空间天然兼容，论文采用相机空间右手坐标系：$+x$ 向右、$+y$ 向上、$+z$ 朝向相机外。

**符号说明**:
- $(x,y,z)$: 单位表面法向量。
- facing left $(-1,0,0)$: 编码为偏粉红/红色。
- facing up $(0,1,0)$: 编码为浅绿色。
- facing camera $(0,0,1)$: 编码为浅蓝/紫色。

---

## 关键图表

### Figure 1: Overview / 系统概览

![[VisionBanana_fig1_overview.png]]

**说明**: 展示从 [[Nano Banana Pro]] 到 [[Vision Banana]] 的转化：模型仍然生成 RGB 图像，但这些图像遵循可解码的视觉任务可视化协议。

### Figure 2: Semantic Segmentation / 语义分割

![[VisionBanana_fig2_semantic_segmentation.png]]

**说明**: [[Vision Banana]] 可理解多种 prompt 写法，包括自然语言颜色、RGB tuple、JSON-like mapping 和 hex code，并能分割细粒度文本描述对象。

### Figure 3: Instance Segmentation / 实例分割

![[VisionBanana_fig3_instance_segmentation.png]]

**说明**: 对未知实例数的问题，论文采用 per-class inference，让模型给同类不同实例动态分配颜色，再用颜色聚类解码 mask。

### Figure 4: Referring Expression Segmentation / 指代表达分割

![[VisionBanana_fig4_referring_segmentation.png]]

**说明**: 展示外观、动作、非常规用途、多语言文字等指代表达，说明生成式预训练带来的语义和关系理解能力。

### Figure 5: Depth-RGB Bijection / 深度颜色双射

![[VisionBanana_fig5_depth_rgb_bijection.png]]

**说明**: 米制深度先经过 [[Power Transform]] 压缩，再沿 RGB cube 边插值。该映射可逆，因此可把生成的 depth visualization 解码回真实深度。

### Figure 6: Metric Depth Demo / 米制深度可视化

![[VisionBanana_fig6_metric_depth_demo.png]]

**说明**: 左两列为输入和生成深度图，右两列为结合相机内参后的 3D 重建视图。相机内参只用于可视化重建，不参与模型预测深度。

### Figure 7: Depth in the Wild / 真实场景深度估计

![[VisionBanana_fig7_depth_in_the_wild.png]]

**说明**: 手机拍摄金阁寺附近照片，Vision Banana 在绿色星标处预测 13.71m，Google Maps 测量为 12.87m，AbsRel 约 0.065。

### Figure 8: Surface Normal / 表面法向估计

![[VisionBanana_fig8_surface_normal.png]]

**说明**: 与 [[Lotus-2]] 比较，Vision Banana 的法向图在细节上更清晰，尽管部分 outdoor benchmark 的数值并非绝对最优。

### Figure 9: Text-to-Image Generation / 文生图保持能力

![[VisionBanana_fig9_text_to_image.png]]

**说明**: 追加视觉任务 instruction tuning 后，Vision Banana 与 Nano Banana Pro 的文生图质量相近。

### Figure 10: Image Editing / 图像编辑保持能力

![[VisionBanana_fig10_image_editing.png]]

**说明**: 在 ImgEdit 样本上，Vision Banana 的编辑能力基本保留，但 win rate 47.8% 略低于 Nano Banana Pro。

### Table 1: 总体性能

| 能力 | Benchmark / Metric | Vision Banana | Best Counterpart |
|------|--------------------|----------------|------------------|
| Referring segmentation | RefCOCOg UMD val cIoU ↑ | **0.738** | 0.734 SAM3 Agent |
| Referring segmentation | ReasonSeg val gIoU ↑ | **0.793** | 0.770 SAM3 Agent |
| Semantic segmentation | Cityscapes val mIoU ↑ | **0.699** | 0.652 SAM3 |
| Instance segmentation | SA-Co/Gold pmF1 ↑ | 0.540* | **0.552 DINO-X** |
| Metric depth | Avg. 4 datasets δ1 ↑ | **0.929** | 0.918 Depth Anything 3 |
| Surface normal | Avg. 4 datasets mean angle error ↓ | **18.928** | 19.642 Lotus-2 |
| Text-to-image | GenAI-Bench win rate ↑ | **53.5%** | 46.5% Nano Banana Pro |
| Image editing | ImgEdit win rate ↑ | 47.8% | **52.2% Nano Banana Pro** |

**说明**: Vision Banana 在 2D/3D 理解任务上超过或接近专用模型，同时保持与基座生成器相近的生成能力。

### Table 2: 分割任务细表

| Dataset | Setting | Metric | Vision Banana | 关键对比 |
|---------|---------|--------|----------------|----------|
| Cityscapes val | Zero-shot transfer | mIoU ↑ | **0.699** | SAM 3 0.652 |
| SA-Co/Gold | Zero-shot transfer | pmF1 ↑ | 0.540* | DINO-X 0.552, Gemini 2.5 0.461 |
| RefCOCOg val (U) | Zero-shot transfer | cIoU ↑ | **0.738** | SAM 3 + Gemini 2.5 Pro 0.734 |
| ReasonSeg val | Zero-shot transfer | gIoU ↑ | **0.793** | SAM 3 Agent + Gemini 2.5 Pro 0.770 |

**关键发现**: semantic/referring 分割很强，instance segmentation 仍略弱于 DINO-X 和非 zero-shot 的 SAM 3。

### Table 3: Monocular Metric Depth Estimation

| Dataset | Metric | Vision Banana | 备注 |
|---------|--------|----------------|------|
| Average | δ1 ↑ | **0.882** | 高于 UniK3D 0.823、MoGe-2 0.802、Depth Pro 0.715 |
| NYU | δ1 ↑ | 0.948 | 略低于若干 specialist |
| iBims1 | δ1 ↑ | **0.934** | 表现强 |
| ETH3D | δ1 ↑ | **0.935** | 高于 Depth Anything V3 0.917 |
| DIODE-Indoor | δ1 ↑ | **0.917** | 高于其他列出方法 |
| KITTI | δ1 ↑ | 0.915 | 低于 Depth Anything V3 0.953 |
| nuScenes | δ1 ↑ | 0.643 | 低于部分方法 |

**说明**: 最大卖点是不用相机内参参与训练或推理，仅从视觉线索和生成式世界知识推断绝对尺度。

### Table 4: Surface Normal Estimation

| Method | Indoor Avg mean ↓ | Indoor Avg median ↓ | VKitti mean ↓ | VKitti median ↓ |
|--------|-------------------|---------------------|---------------|-----------------|
| Marigold | 19.606 | 11.828 | - | - |
| DSINE | 17.017 | 10.190 | **28.9** | **9.9** |
| StableNormal | 17.168 | 10.028 | - | - |
| Lotus-2-Normal | 16.558 | - | 28.894 | 9.677 |
| Vision Banana | **15.549** | **9.300** | 29.063 | 10.699 |

**说明**: 室内平均最强，outdoor 数值接近 SOTA，但作者强调其仍保持 strict zero-shot transfer。

---

## 实验

### 数据集

| 数据集 | 特点 | 用途 |
|--------|------|------|
| Cityscapes | 城市场景 19 类语义分割 | 语义分割评测 |
| SA-Co/Gold | segmentation-anything 风格实例/开放集分割 | 实例分割评测 |
| RefCOCOg UMD | 自然语言 referring expression | 指代表达分割评测 |
| ReasonSeg | 需要语言推理的分割 | 指代表达/推理分割评测 |
| NYU / iBims1 / ETH3D / DIODE / KITTI / nuScenes | 单目深度 benchmark | metric depth 评测 |
| NYUv2 / DIODE-indoor / ScanNet / VKitti | 表面法向 benchmark | surface normal 评测 |
| GenAI-Bench | 文生图人评 | 生成能力保持评估 |
| ImgEdit | 图像编辑人评 | 编辑能力保持评估 |

### 实现细节

- **Backbone**: [[Nano Banana Pro]]。
- **训练方式**: 在原始生成数据混合中加入低比例视觉任务数据进行 [[Instruction Tuning]]。
- **2D 训练数据**: web-crawled 2D 图像的内部模型标注。
- **3D 训练数据**: 渲染引擎生成的合成数据。
- **评测约束**: 作者声明不使用各评测 benchmark 的训练集。
- **优化器 / batch size / 训练步数 / 硬件**: 未披露。

### 可视化结果

可视化显示 Vision Banana 对 prompt 格式相当鲁棒，能处理自然语言、JSON-like mapping、RGB tuple 和 hex color；在 depth 和 normal 上细节也较好。比较值得注意的是，它生成的结果不是传统 logits 或连续张量，而是“可解码图像”，因此格式稳定性本身就是能力的一部分。

---

## 批判性思考

### 优点

1. **范式清晰**: 把视觉理解统一为图像生成，接口设计非常干净。
2. **结果强**: 在多个 zero-shot transfer benchmark 上超过或接近专用 SOTA。
3. **保持生成能力**: 低比例视觉数据混合避免明显 catastrophic forgetting。
4. **跨任务迁移自然**: semantic/instance/referring segmentation 之间能复用语言理解与视觉 grounding 能力。

### 局限性

1. **不可复现性强**: [[Nano Banana Pro]]、训练数据、训练比例、训练步数和模型规模均未完整公开。
2. **成本高**: 作者承认图像生成器推理开销远高于轻量专用模型。
3. **RGB 解码接口有工程摩擦**: 颜色偏差、压缩、插值和采样随机性都可能影响定量解码。
4. **实例分割仍有短板**: SA-Co/Gold 上略低于 DINO-X，且明显低于非 zero-shot 的 SAM 3。
5. **预训练数据边界不透明**: zero-shot transfer 的严格性仍受大规模生成预训练数据可审计性影响。

### 潜在改进方向

1. 扩展更多视觉任务，如 optical flow、pose、tracking、video segmentation 和 multi-view reconstruction。
2. 研究更稳定、更高效的可解码输出协议，降低 RGB 采样误差。
3. 将 [[Foundational Vision Models|基础视觉模型]] 与 [[Large Language Model|大语言模型]] 更紧密结合，增强跨模态推理。
4. 做蒸馏或加速，把生成式通用模型的能力迁移到低成本部署模型。

### 可复现性评估

- [ ] 代码开源
- [ ] 预训练模型
- [ ] 训练细节完整
- [x] 数据集/评测名称披露
- [x] arXiv 源码包含论文图像素材

---

## 关联笔记

### 基于

- [[Nano Banana Pro]]: 作为基座图像生成器。
- [[Generative Vision Pretraining]]: 论文的核心预训练范式假设。
- [[Instruction Tuning]]: 用于把生成模型对齐到可解码视觉任务格式。

### 对比

- [[Segment Anything Model 3]]: 分割领域强 baseline。
- [[Depth Anything V3]]: metric depth 领域强 baseline。
- [[Lotus-2]]: surface normal / geometry diffusion baseline。
- [[DINO-X]]: instance segmentation 上接近或略强的 zero-shot baseline。

### 方法相关

- [[RGB Output Parameterization]]: 核心统一接口。
- [[Zero-Shot Transfer]]: 主要评测设定。
- [[Monocular Metric Depth Estimation]]: 重要 3D 任务。
- [[Surface Normal Estimation]]: 重要 3D 任务。
- [[Referring Expression Segmentation]]: 体现语言 grounding 能力。

---

## 速查卡片

> [!summary] Image Generators are Generalist Vision Learners
> - **核心**: 强图像生成器通过轻量 instruction tuning 可成为通用视觉理解模型。
> - **方法**: 把分割、深度、法向量等任务统一编码为可解码 RGB 图像生成。
> - **结果**: 在 semantic/referring segmentation、metric depth、surface normal 上达到或接近 SOTA。
> - **代码**: 未开源；项目页 https://vision-banana.github.io/

---

*笔记创建时间: 2026-04-26 12:10 Asia/Shanghai*
