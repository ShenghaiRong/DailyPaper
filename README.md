# DailyPaper

> 每日论文推荐和笔记自动化 Obsidian Vault

自动抓取 arXiv/HuggingFace 新论文，按研究方向筛选，生成推荐列表并保存到 Obsidian。

## 📁 目录结构

```
DailyPapers/          # 每日论文推荐
  └── YYYY-MM-DD-论文推荐.md

论文笔记/              # 单篇论文完整笔记
  ├── _概念/           # 技术概念库
  │   └── 0-待分类/
  └── _待整理/         # 待分类的笔记
```

## 🔬 研究方向

当前关注的领域：

- World Model
- Diffusion Model
- Embodied AI
- 3D/4D Gaussian Splatting
- Sim-to-Real / Sim2Real
- Robot Simulation
- Autonomous Driving
- Self-Driving

## 🤖 自动化流程

每天早上 9:00 自动：

1. 抓取新论文（arXiv + HuggingFace）
2. 按关键词打分筛选
3. 生成推荐列表（必读 / 值得看 / 可跳过）
4. 自动提交并推送到 GitHub
5. 微信推送摘要

## 📚 技术栈

- **技能**: dailypaper-skills
- **源**: arXiv, HuggingFace Daily/Trending
- **存储**: Obsidian Vault + GitHub
- **自动化**: OpenClaw Cron

---

> 自动生成于 2026-03-27 | Powered by OpenClaw + dailypaper-skills