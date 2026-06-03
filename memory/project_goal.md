# 项目目标（用户 2026-05-31 定）

把这个 Block Blast 类游戏的"运气 vs 技能"研究**做成一个有一定创新性的小项目**，并**推送到 GitHub**。

## 用户原话要点
- 写成"具有一定创新性的小项目"：关于它的解法、各个方面，用一些**数学方法**。
- 先**搜网上现有策略/研究**，摸清现状。
- **搞一点小创意/创新点**出来（不是复述已有 solver）。
- 写完后**得出一些结论**。
- **推送到 GitHub**。

## 已有素材（可直接进 repo）
- pieces.py(38方块目录) / scoring.py(非线性计分) / sim.py(引擎+方差分解)
- fast.py(bitboard+beam-strong+lookahead) / compare.py(配对CRN) / ladder.py / run_lookahead.py
- memory/ 下结论：方差分解、技能阶梯交叉点、收敛、oracle 缺口

## 候选创新角度（待筛选）
1. **"运气天花板"= oracle 缺口 = 信息价值(value of hindsight)** 的量化框架——把单人随机游戏的
   luck/skill 用 in-game value-of-information 表达，可能是相对新颖的提法。
2. **技能阶梯 + 缺口/σ 交叉点**：给"多少分以下技能主导/以上运气主导"一个可操作定义。
3. **碎片化/最大空矩形 的相变(percolation)视角**：棋盘"死亡"的临界密度。
4. **combo/all-clear 非线性奖励如何改变 luck/skill 配比** 的实证。

## 长期研究方向（2026-06-01 用户提出，待立项）
**逆问题：胜率可设计性 / 游戏公司难度调控（DDA）。** 现有框架是正问题（均匀发牌→luck/skill 各几成）；
逆过来：**给定目标胜率/存活率 W\***，**设计发牌概率 θ（38 块的非均匀分布）+ 计分参数**去命中它。
- 站在游戏公司立场：胜率是可调的产品指标（留存/付费曲线）。旋钮 = 各方块出现概率、bag/补牌机制、计分权重。
- 形式化：forward W(θ,π) 对玩家策略 π（用现有 strong/seer 当不同技能档位）可测；inverse = 求 θ 命中 W\*。
  38 维中等规模 → 灵敏度分析 ∂W/∂θ_i（哪块概率最能搬动胜率）/ grid / 贝叶斯优化 / 通过模拟器的 REINFORCE 梯度。
- **与本项目主线的耦合（卖点）**：调难度有两条路——改**存活运气**(抬高 Burgiel 杀手序列频率，玩家觉得"被坑")
  vs 改**技能上限**(让局面更需要规划)。luck/skill 分解正是审计"这个难度旋钮诚实(改技能) 还是操纵(改运气)"的工具。
  → "胜率作为可设计目标 + 难度旋钮的 luck/skill 归因" 可能是比正问题更有产品价值的新颖提法。
- 注意陷阱：θ 非均匀会**同时**改 survival luck 和 skill ceiling（耦合，不能只看一个）；需在固定 π 档位上分别报。

## 交付 — 已完成 (2026-05-31)
- **私有 repo**: https://github.com/sher-shen/block-blast-luck-vs-skill (PRIVATE)
- 创新主线(用户选)：**运气=信息价值(oracle 缺口)** = "value of hindsight"。
- 新增交付物：README.md(完整 writeup) / experiments.py(一键出 results.json) /
  plots.py(零依赖 SVG) / figures/*.svg / LICENSE(MIT) / .gitignore(排除 memory/)。
- 最终数字(200/24 种子,部分线性版 legacy)：技能地板113×；熟练间 ANOVA 技能34/运气34/交互32；交叉点≈3408。
- ⚠️ **"oracle 缺口≈78%" 已撤回**(ratio-of-means 口径坏掉)→改两通道：存活 hazard(点估计7e-5/轮,Poisson上界3.3e-4) + EVPI 信息占比 **57–69%**(随T,见 [[oracle_immortality_reframe]])。
- ⚠️ **目标已升级为双目标**(2026-06-01,见根 GOAL.md)：①最强可玩策略(②a 8×8 RL) + ②运气/技能分解。本文件历史数字以 GOAL.md / README 为准。
- git 身份：user.name=sher-shen, email=s1248830519@gmail.com (仓库本地配置)。
