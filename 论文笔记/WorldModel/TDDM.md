---
title: "Temporally Decoupled Diffusion Planning for Autonomous Driving"
method_name: "TDDM"
authors: [Xiang Li, Bikun Wang, John Zhang, Jianjun Wang]
year: 2026
venue: arXiv
tags: [autonomous-driving, motion-planning, diffusion-model, temporal-decoupling, trajectory-generation]
arxiv_url: https://arxiv.org/abs/2603.25462
created: 2026-03-28
---

# Temporally Decoupled Diffusion Planning for Autonomous Driving

## Abstract

Motion planning in dynamic urban environments requires balancing immediate safety with long-term goals. While [[Diffusion Models]] effectively capture [[Multi-modal Decision-making]], existing approaches treat trajectories as monolithic entities, overlooking heterogeneous [[Temporal Dependencies]] where near-term plans are constrained by instantaneous dynamics and far-term plans by navigational goals. To address this, we propose [[Temporally Decoupled Diffusion Model|TDDM]], which reformulates trajectory generation via a [[Noise-as-mask]] paradigm. By partitioning trajectories into segments with independent noise levels, we implicitly treat high noise as information voids and weak noise as contextual cues. This compels the model to reconstruct corrupted near-term states by leveraging internal correlations with better-preserved temporal contexts. Architecturally, we introduce [[Temporally Decoupled Adaptive Layer Normalization|TD-AdaLN]] to inject segment-specific timesteps. During inference, our [[Asymmetric Temporal Classifier-Free Guidance|Asymmetric Temporal CFG]] utilizes weakly noised far-term priors to guide immediate path generation. Evaluations on the nuPlan benchmark show TDDM approaches or exceeds state-of-the-art baselines, particularly excelling in the challenging Test14-hard subset.

## Introduction

Motion planning is one of the core technical challenges in [[Autonomous Driving]]. The objective is generating safe, comfortable, and human-like trajectories in dynamically changing and uncertain environments.

### Traditional vs Learning-based Approaches

**Traditional planning methods:**
- Search-based approaches
- Optimization-based approaches
- Rely on precise environmental models and complex heuristic rules
- Struggle with highly dynamic and interactive nature of complex urban scenarios

**Learning-based planning:**
- [[Imitation Learning]] (IL) learns driving policies from large-scale data
- [[End-to-End]] (E2E) methods directly learn mapping from perception to control
- **Two core challenges:**
  1. [[Distribution Shift]] - state distribution during testing differs from training, leading to compounding errors
  2. [[Multi-modal Decision-making]] - human drivers might make several equally valid and safe decisions, whereas single regression yields overly conservative policies

### Diffusion Models for Trajectory Planning

Diffusion models have been introduced into trajectory planning due to their powerful ability to model complex data distributions. By redefining planning as a conditional trajectory generation problem, diffusion models can start from random noise and generate diverse, plausible trajectories through a denoising process guided by scene context and kinematic constraints.

However, existing diffusion-based methods treat the entire trajectory as a monolithic, indivisible entity, ignoring:
- Near-term segments: more dependent on vehicle's precise current state and instantaneous dynamics
- Far-term segments: more focused on achieving long-term navigational goals

This simplification restricts the model's ability to generate trajectories that optimally balance short-term safety with long-term consistency.

## Main Contributions

1. **Novel temporally-decoupled training paradigm:** Partition trajectory into multiple temporal segments and apply different levels of noise independently to each segment during training. This randomized temporal masking mechanism significantly enhances the model's ability to capture complex temporal correlations.

2. **Asymmetric temporal guidance strategy for inference:** Extend [[Classifier-Free Guidance|CFG]] to the temporal dimension, using weakly noised far-term priors as a condition to guide near-term plan generation. This effectively frames inference as a goal-directed trajectory completion task, enhancing long-term consistency.

3. **Efficient model architecture:** Based on [[Diffusion Transformer|DiT]] framework, incorporates [[Temporally Decoupled Adaptive Layer Normalization|TD-AdaLN]] module for independent injection of diffusion timestep information for each temporal segment.

4. **Strong experimental performance:** Comprehensive evaluations on nuPlan benchmark demonstrate superiority, particularly in challenging long-tail scenarios where TDDM exhibits exceptional robustness and planning consistency.

## Related Work

### Imitation-based Planning

Evolution of architectures:
- Early approaches: Simple [[Convolutional Neural Networks|CNN]]
- ChauffeurNet: Incorporated [[Recurrent Neural Networks|RNN]]
- Recent methods: [[Transformer]] architecture (e.g., planTF)
- Hierarchical approaches: PLUTO (decouples lateral and longitudinal control)
- Unified frameworks: UniAD (integrates perception, prediction, and planning)

Persistent challenge: effectively capturing multi-modal nature of driving decisions.

### Diffusion Models for Trajectory Planning

- **MotionDiffuser:** Among first to apply diffusion to multi-agent motion forecasting
- **Diffusion-ES:** Combines diffusion models with evolutionary strategies for robustness
- **DiffusionDrive:** Initiates denoising from learned anchors, accelerating inference
- **DiffusionPlanner:** Joint prediction problem for all agents with style-conditioned generation

Limitation: Treat entire trajectory as monolithic entity, overlooking heterogeneous temporal dependencies.

### Decoupled Modeling in Sequence Generation

Inspiration from video generation:
- **Diffusion Forcing:** Applies noise independently to video frames, forcing model to learn fine-grained causal relationships
- **CausVid/DFoT:** Explores efficient causal structures for generation speed and interactivity

Key insight: By decomposing a sequence and explicitly modeling inter-unit dependencies, it's possible to significantly improve generation quality and temporal coherence.

## Preliminary

### Conditional Diffusion Model

A conditional diffusion model learns to reverse a gradual noising process to generate data samples conditioned on some context $c$.

#### Forward Process

The forward process is a fixed Markov chain that incrementally adds Gaussian noise to original data $\tau^{0}$:

$$
q(\tau^{i}\mid\tau^{0})=\mathcal{N}\left(\tau^{i};\sqrt{\bar{\alpha}^{i}}\,\tau^{0},(1-\bar{\alpha}^{i})\mathbf{I}\right),
$$

**Symbol Description:**
- $\tau^{i}$: Noised data at timestep $i\in[0,1]$
- $\{\bar{\alpha}^{i}\}$: Predefined noise schedule
- $\mathcal{N}$: Gaussian distribution
- $\mathbf{I}$: Identity matrix

#### Reverse Process

The reverse process involves training a neural network $f_{\theta}(\tau^{i},c,i)$ to denoise $\tau^{i}$ back towards $\tau^{0}$ given the condition $c$:

$$
p_{\theta}(\tau^{0}\mid c)=\int p(\tau^{T})\prod_{t=1}^{T}p_{\theta}(\tau^{t-1}\mid\tau^{t},c)d\tau^{1:T}.
$$

### Anchor-based Trajectory Vocabulary

To structure the motion planning problem, we discretize the continuous action space into a predefined trajectory vocabulary:

$$
V=\{v_{i}\}_{i=1}^{M}
$$

- Created by applying [[k-means clustering]] to large-scale dataset of expert-driven trajectories (e.g., nuPlan)
- Each cluster centroid $v_{i}\in V$ serves as a trajectory anchor, representing a distinct, kinematically feasible driving maneuver
- Each anchor $\tau$ is composed of a sequence of $T_{h}$ waypoints:
  - $\tau=\{(x_{t},y_{t},\phi_{t})\}_{t=1}^{T_{h}}$
  - Captures vehicle's position and heading over planning horizon

### Problem Formulation

We formulate motion planning as a conditional generation task. The core idea is to learn a denoising model that can refine a set of noisy trajectory anchors into a final multimodal trajectory distribution, conditioned on scene context $c$.

**Tokenization:**
- Each trajectory anchor $v\in V$ is segmented into $N$ temporal segments: $\{\tau_{1}^{0},\tau_{2}^{0},\dots,\tau_{N}^{0}\}$

**Independent Noising:**
- For the $n$-th segment $\tau_{n}^{0}$, sample independent diffusion timestep $i_{n}\in[0,1]$ and standard Gaussian noise $\epsilon_{n}\sim\mathcal{N}(0,\mathbf{I})$
- Resulting noised segment:

$$
\tau_{n}^{i_{n}}=\sqrt{\bar{\alpha}^{i_{n}}}\tau_{n}^{0}+\sqrt{1-\bar{\alpha}^{i_{n}}}\epsilon_{n}.
$$

This independent noising forces the model to learn complex temporal dependencies between segments, rather than relying on uniform noise patterns.

**Overall Objective:**
Learn weights $\theta$ of denoising network $f_{\theta}$ that takes scene context $c$ and set of noised anchors as input, and outputs refined set of trajectories $\{\hat{\tau}_{k}\}$ along with confidence scores $\{\hat{s}_{k}\}$:

$$
\{\hat{s}_{k},\hat{\tau}_{k}\}_{k=1}^{M}=f_{\theta}(\{\{\tau_{k,n}^{i_{n}}\}_{n=1}^{N}\}_{k=1}^{M},c).
$$

Essentially, the model learns to jointly denoise all anchors and predict which one represents the optimal plan for the given context.

## Approach

### Temporally Decoupled Diffusion Model

Inspired by recent video generation advancements (Diffusion Forcing, CausVid, DFoT), we leverage structural commonalities with trajectory generation. We reformulate trajectory generation as a denoising process decoupled in the time dimension.

**Core idea:** Partition a complete trajectory into multiple temporal tokens and apply independent random noise to these tokens during training. This compels the model to:
1. Learn kinematic smoothness within each token
2. Leverage global context to understand and reconstruct complex temporal correlations between tokens

#### Architecture Challenges

Two principal challenges:
1. Accommodate application of distinct diffusion timestep encodings to each temporally decoupled segment
2. Ensure kinematic consistency across entire planning horizon

#### Input Processing

Trajectory is decomposed into $N$ temporal segments, conceptually clustered into $G$ macro-groups:
- $N$: Number of temporal segments (flexible)
- $G$: Number of macro-groups (dictates component instantiation)

**Position Encoding:**
- Standard group-specific positional encodings applied to each segment
- Processed through shared MLP projection layer $\mathcal{F}_{\text{pre}}$
- Group features concatenated along channel dimension to form unified feature tensor $h$:

$$
h_{g}=\mathcal{F}_{\text{pre}}(\text{pos}(\tau_{g}));\quad h=\text{Concat}(h_{1},h_{2},\dots,h_{G}).
$$

#### Temporally Decoupled Adaptive Layer Normalization (TD-AdaLN)

For each macro-group $g\in\{1,\dots,G\}$, a conditional vector $y_{g}$ is constructed by combining group-specific diffusion timestep encoding $t_{g}$ with shared navigation encoding:

$$
y_{g}=\mathcal{F}_{\text{time}}(t_{g})+\mathcal{F}_{\text{navi}}(\text{navi}).
$$

**Symbol Description:**
- $t_{g}$: Group-specific diffusion timestep
- $\text{navi}\in\mathbb{R}^{(K\times P)\times D_{\text{route}}}$: Navigation information from nuPlan (K route lanes, P points, D_route coordinate features)
- $\mathcal{F}_{\text{time}}$, $\mathcal{F}_{\text{navi}}$: Encoding networks

Within each [[Transformer]] block, these conditional vectors generate segment-specific modulation parameters:

$$
\text{params}_{g}=\mathcal{F}_{\text{adaLN}}(y_{g}),
$$

which are concatenated along channel dimension to form complete modulation tensor and applied to main feature tensor $h$ to conditionally modulate its normalization statistics.

#### Backbone

- Series of [[DiT]] blocks with [[Self-Attention]] and [[Cross-Attention]] modules
- Self-Attention: Facilitates information fusion among different temporal segments, enforcing internal consistency
- Cross-Attention: Integrates unified external scene information $c$ (nearest neighbors, map elements, static obstacles)
- Dedicated feed-forward network (FFN): Predicts confidence score for whole trajectory
- Segments decoded independently by group-specific FFNs

![Figure 2: Overview of temporally decoupled diffusion model](https://arxiv.org/html/2603.25462v1/2603.25462v1/architecture.png)

### Asymmetric Temporal CFG

Temporal decoupling enables unprecedented flexibility in reference by assigning distinct denoising starting points to different temporal segments.

#### Two Parallel Paths at Each Denoising Step

**1. Unconditional Path:**
- Represents original full-sequence diffusion mode
- Start from trajectory vocabulary composed entirely of Gaussian noise without temporal decoupling
- Same noise added to all segments, consistent noise level at each denoising step

**2. Conditional Path:**
- Defines deterministic assumption about long-term future
- Divide trajectory segments: near-term plan (first $N/2$ segments) and far-term plan (last $N/2$ segments)
- At each inference step, construct asymmetric, mixed-noise-level input:
  - **Near-term:** Start from full noise
  - **Far-term:** Always kept as weakly noised original prototype (anchor) form, corresponding to diffusion timestep of 0.001
- Poses a completion task: given a future target, generate optimal current path

**3. CFG Fusion:**
Fuse outputs from both paths according to [[Classifier-Free Guidance|CFG]] formula:

$$
\hat{\tau}_{g,\text{final}}^{0}=\hat{\tau}_{g,\text{uncond}}^{0}+w\cdot\left(\hat{\tau}_{g,\text{cond}}^{0}-\hat{\tau}_{g,\text{uncond}}^{0}\right),
$$

**Symbol Description:**
- $w$: Guidance scale controlling adherence strength to long-term target
  - $w>1$: Enforces stronger conditional guidance
  - $0<w<1$: Weakens guidance

Use reparameterization trick to directly predict clean trajectory.

![Figure 3: Pipeline of Asymmetric Temporal CFG](https://arxiv.org/html/2603.25462v1/2603.25462v1/sample.png)

#### Continuity Loss

To ensure kinematic consistency across the entire planning horizon, a continuity loss is applied:

$$
\mathcal{L}_{\text{cont}}=\sum_{g=1}^{G}\gamma\cdot\left(\|\text{start}_{g}(\cdot)-\text{end}_{g-1}(\cdot)\|_{2}+\|\text{end}_{g}(\cdot)-\text{start}_{g+1}(\cdot)\|_{2}\right),
$$

**Symbol Description:**
- $\text{end}_{g}(\cdot)$ and $\text{start}_{g}(\cdot)$: Last and first waypoints of a group-specific segment, respectively
- $\gamma$: Hyperparameter to weigh continuity constraint

### Training

The model is trained using the temporally-decoupled paradigm with independent noise applied to each trajectory segment.

## Experiment

### Dataset and Benchmarks

**Dataset:** [[nuPlan]]
- Large-scale multi-modal dataset containing $\sim$1,500 hours of real-world driving data from four cities
- Features diverse urban scenarios: merging, roundabouts, complex agent interactions

**Evaluation Settings:**
- Non-reactive (NR): Other agents merely replay historical behaviors
- Reactive (R): Other agents dynamically react to ego vehicle

**Benchmarks:**
1. **Val14:** 1,118 regular validation scenarios
2. **Test14-hard:** 272 long-tail, high-risk scenarios
3. **Test14-random:** Over 200 randomly sampled scenarios

**Baselines:**
- Rule-based: IDM, PDM (PDM-Closed, PDM-Open, PDM-Hybrid)
- Learning-based:
  - UrbanDriver (policy-gradient method)
  - GameFormer (game-theory-inspired)
  - PlanTF (Transformer-based)
  - PLUTO (Transformer-based)
  - DiffusionPlanner (diffusion-based)

### Implementation Details

#### Architecture and Training
- Aligned with Diffusion Planner for fair comparison
- Modified diffusion backbone into multi-modal, anchor-based architecture
- 20 trajectory anchors derived via [[k-means clustering]]
- Trained for 500 epochs on 200k-scenario nuPlan subset
- 4 NVIDIA 3080 GPUs (batch size 320, AdamW optimizer, learning rate $5\times 10^{-4}$)

#### Inference
- Fast inference with only 2 denoising steps via DPM-Solver++
- VP noise schedule
- Highest-confidence trajectory propagated at each step
- CFG accelerated through parallel batching
- Outputs 8 seconds trajectory at 10 Hz

### Main Results

#### Quantitative Results

![Table 1: Comparison of planning performance on nuPlan](https://arxiv.org/html/2603.25462v1/2603.25462v1/2603.25462v1.html)

**Key Findings:**
- TDDM achieves competitive results using only 200k scenarios (vs. Diffusion Planner requiring 1M)
- In non-reactive (NR) evaluation:
  - Val14: TDDM scores 89.81 (on par with Diffusion Planner's 89.76)
  - Test14-hard: TDDM scores **77.95** (significantly outperforms Diffusion Planner's 75.67)
  - Test14-random: TDDM scores **90.4** (leading position)

**Superiority in challenging scenarios:**
- TDDM demonstrates exceptional robustness and planning consistency on difficult benchmarks
- Particularly excels in long-tail, high-risk scenarios
- Effectiveness of temporal decoupling mechanism evident

#### Qualitative Comparison

![Figure 4: Qualitative comparison](https://arxiv.org/html/2603.25462v1/2603.25462v1/2603.25462v1.html)

**Challenging scenarios where baseline fails:**
1. **Narrow right turn:** Baseline Diffusion Planner generates trajectory that collides with parked bus
2. **Left turn with obstacle:** Baseline makes critical misjudgment, stopping unnecessarily behind parked vehicle

**TDDM performance:**
- Successfully navigates both situations
- Produces smooth, safe, and decisive maneuvers
- Asymmetric temporal guidance effectively prevents myopic, inconsistent decisions

### Ablation Study

![Table 2: Ablation study on nuPlan Test14-hard](https://arxiv.org/html/2603.25462v1/2603.25462v1/2603.25462v1.html)

#### Component Analysis

| ID | Configuration | Score |
|-----|--------------|-------|
| 1 (Baseline) | Anchor-based Diffusion Planner | 75.91 |
| 2 | Trajectory Tokenization only | 76.60 |
| 3 | TD-AdaLN without independent noise | 74.94 |
| 4 | Independent noise without TD-AdaLN | 73.76 |
| 5 | Both TD-AdaLN + independent noise | 76.88 |
| 6 | + Asymmetric Temporal CFG | **77.95** |

**Key Insights:**
- TD-AdaLN module is essential for handling segment-specific timestep information
- Demonstrates critical synergy between model architecture and training paradigm
- Both components (TD-AdaLN + independent noise) must be aligned to enhance learning

#### Trajectory Tokenization (Optimal Granularity)

| N (Tokens) | Score | Note |
|--------------|-------|------|
| 2 | 76.60 | Too coarse |
| 4 | **77.95** | **Optimal** |
| 8 | 73.76 | Too fine |

**Finding:** Moderate granularity (N=4 tokens, 2-second segment length) strikes optimal balance between capturing complex temporal dynamics and maintaining long-term kinematic consistency.

#### Classifier-Free Guidance (Optimal Scale)

| w (Guidance Scale) | Score | Note |
|--------------------|-------|------|
| < 1.25 | 76.88 | Insufficient guidance |
| **1.25** | **77.95** | **Optimal** |
| > 1.25 | 76.60 | Excessive guidance |

**Finding:** Optimal scale (w=1.25) strikes balance between enforcing long-term goal consistency and preserving short-term reactive flexibility.

## Conclusion

TDDM introduces a novel framework for autonomous driving motion planning that addresses limitations inherent in monolithic trajectory generation.

### Key Contributions Summary

1. **Temporally-decoupled training scheme** with independent noise - learns to capture heterogeneous dependencies across planning horizon
2. **TD-AdaLN architecture** - provides architectural support for segment-specific timesteps
3. **Asymmetric Temporal CFG** - uses weakly noised far-term priors to guide near-term generation

**Synergy:** These components work together to produce trajectories that are both reactive to immediate conditions and consistent with long-term goals.

### Performance Summary

Experiments on nuPlan benchmark demonstrate that TDDM **approaches or exceeds** state-of-the-art learning-based methods, showcasing exceptional robustness and coherence on challenging Test14-hard subset.

### Limitations and Future Work

**Current limitations:**
- Efficacy in fully reactive, closed-loop simulations is an area for improvement

**Future directions:**
1. Enhancing agent interaction, potentially extending CFG with game-theoretic principles
2. Removing reliance on predefined anchor set by exploring autoregressive generation from historical trajectories
3. Further exploiting temporal token structure for more explicit and fine-grained trajectory guidance

## Code

Available at: https://github.com/wodlx/TDDM

## Key Concepts

- [[Autonomous Driving]]
- [[Motion Planning]]
- [[Diffusion Models]]
- [[Multi-modal Decision-making]]
- [[Temporal Dependencies]]
- [[Imitation Learning]]
- [[End-to-End]]
- [[Distribution Shift]]
- [[Classifier-Free Guidance|CFG]]
- [[Noise-as-mask]]
- [[Temporally Decoupled Adaptive Layer Normalization|TD-AdaLN]]
- [[Asymmetric Temporal Classifier-Free Guidance]]
- [[Anchor-based Trajectory Vocabulary]]
- [[k-means clustering]]
- [[Trajectory Tokenization]]
- [[Transformer]]
- [[Diffusion Transformer|DiT]]
- [[Self-Attention]]
- [[Cross-Attention]]
- [[nuPlan]]
- [[Continuity Loss]]
