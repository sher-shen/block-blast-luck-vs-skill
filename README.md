# Luck as the Value of Hindsight
### 用"信息价值"定量分解 Block Blast 类游戏的运气与技能

> **一句话**：在 8×8 Block Blast 这类"单人 + 随机发块 + 生存型"游戏里，我们把**运气**操作化为
> 一个可计算的量——**一个开了"上帝视角"（提前知道未来方块）的玩家，比只能看当前的在线最优玩家
> 多拿的那部分分**。这正是玩家**缺失的信息的价值（value of hindsight）**。再用方差分解、技能阶梯、
> 搜索收敛三条独立证据互相印证，给出"这游戏到底几分靠运气"的量化答案。

---

## 为什么做这个 / 和已有工作的区别

网上关于 Block Blast 的东西几乎全是两类：**(1) 帮你通关的"解题器"网站**；**(2) 训练一个 RL
智能体把游戏玩好**（如 [RisticDjordje/BlockBlast-Game-AI-Agent](https://github.com/RisticDjordje/BlockBlast-Game-AI-Agent)
的 DQN/PPO）。它们都在问"**怎么玩得更好**"。

而 skill-vs-luck 的学术文献（[Skill vs Chance, arXiv:2410.14363](https://arxiv.org/pdf/2410.14363)、
[Geometry of Games, arXiv:2511.11611](https://arxiv.org/pdf/2511.11611)）几乎全是**多人**博弈与评分系统。

**空白点**：没人把 Block Blast 当成"**运气–技能定量分解**"问题来做，更没人对**单人随机生存型游戏**
用**信息价值**去定义运气。本项目就填这个缝。理论锚点来自一句老话——
[*"luck is nothing more than a lack of information"* (Aleph Insights)](https://alephinsights.com/blog/2016/05/skill-and-luck/)——
我们把它**操作化、算出来**。

---

## 游戏与建模

- 8×8 棋盘，无重力；每轮发 **3 个**方块，必须全放完才刷新；填满整行/整列即消除。
- **方块目录 38 种**（含旋转朝向，均匀等概率，见 [`pieces.py`](pieces.py)）。
- **非线性计分**（[`scoring.py`](scoring.py)）：同时消多行按三角数增长、连击(combo)、清盘(all-clear)均有奖励——
  这类奖励专门奖励"跨手布局"，是分析的关键。
- 引擎两版：可读参考实现 [`sim.py`](sim.py)（列表棋盘）+ 高速 [`fast.py`](fast.py)（64-bit bitboard，
  位运算 can_place / 消行 / 启发式）。

四种玩家（技能由低到高）：`random`(乱放) → `greedy`(逐块贪心) → `strong`(手内 beam 搜索，连击贯穿)
→ `lookahead`(beam 候选 + flat Monte-Carlo rollout，λ=1，CRN 配对)。

---

## 方法（三条独立证据 + 一个新指标）

1. **方差分解（ANOVA）**：同一批随机种子喂给不同玩家，把总分方差拆成
   *技能(玩家主效应)* / *运气(种子主效应)* / *交互*。
2. **技能阶梯**：用 ε-greedy 在 random↔strong 间连续插值造出"技能轴"，
   定义**交叉点** = "到天花板的缺口 = 单局运气波动 σ"处的分数——
   **缺口 > σ → 技能主导；缺口 < σ → 运气主导**。回答"多少分以下靠练、以上靠命"。
3. **搜索收敛**：不断加强玩家，看天花板是否停止上移。若 lookahead ≈ strong，说明已近技能上限，
   剩余方差就是不可消除的运气。
4. **【创新指标】Oracle 缺口 = 信息价值**：让一个 oracle 玩家在 rollout 时使用**真实未来方块**
   （hindsight 上界），其均分与在线最强玩家之差占比 = **运气给技能设的天花板**。

> 方案在动手前先经过一个独立审核 agent 评审（GO-WITH-CHANGES），落实了 7 项修正，其中最关键：
> rollout 基策略必须 ≥ 被改进策略（否则发生 *rollout 回归*）、rollout 用共同随机数(CRN)配对、
> 天花板不能循环自定义。

---

## 结论

![players](figures/fig1_players.svg)
![ladder](figures/fig2_ladder.svg)

数据：38 种方块，方差/阶梯 200 种子，配对/oracle 24 种子（[`results.json`](results.json)）。

**① 技能地板巨大——这是"技能游戏"的一面。**
乱放均分 **57**，会玩（strong）**6453**，差 **≈113×**。光"不犯傻"就决定了两个数量级。

**② 熟练玩家之间，运气≈技能——这是"运气游戏"的一面。**
方差分解（greedy vs strong）：**技能 34% / 运气 34% / 交互 32%**。
而那 32% 交互项是"**只有会玩的人才兑得出的好牌机会**"，本质偏运气。

**③ 分界不是一刀切，而是在很高处。**
技能阶梯显示：分数 **~3400（交叉点）以下几乎全程技能主导**（缺口/σ 从 139 一路降到 ~6，都 >1），
只有逼近天花板才翻成运气主导。**所以"低分=没练好、不是运气差"对绝大多数水平都成立。**

**④ 搜索收敛 + 信息缺口 = 运气的硬度量。**
- *收敛*：`lookahead(strong基) − strong = −1619 ± 1545`（~1 SE 内，统计打平）→ beam-strong 已近
  这一启发式族的技能天花板；加搜索深度收益递减。（对照：greedy 基 rollout `−3039 ± 1530` **更差**，
  正是 rollout 回归；换 strong 基 `+1420 ± 1041` 回血，证实"基策略须 ≥ 被改进者"。）
- *信息缺口*：oracle（开未来牌）≈ **27475** vs 在线 strong ≈ **6066** → **缺口 ≈ 78%**。
  即**在线玩家约八成的"可得分潜力"被'不知道下一手发什么'吃掉了**——这就是运气的量化上界。

**总结**：Block Blast 是一个**"技能定地板、运气定天花板"**的游戏。中低水平拼策略（地板 113×），
顶端拼牌运（信息缺口 ~78%、CV~100%），分界线在很高的位置。它与五子棋（运气=0、有必胜策略）
是不同范式：这里**不存在必胜策略**，连"无限存活"都被
[Burgiel《How to lose at Tetris》](https://www.semanticscholar.org/paper/How-to-lose-at-Tetris-Burgiel/11c12871bfa138fa8bb93a4e5dbcca36c5d214fa)
式的杀手序列否定。

---

## 诚实的局限

- 配对实验 **N=24 偏小**，单局**方差极大（CV~100%）**，故配对差值都带 ±SE，结论按显著性而非点估计读。
- **oracle 是宽松上界**（既开未来牌又用 beam），且被爆炸局长尾拉高，78% 应理解为**运气占比的上界**，
  保守说法是"**过半**"。
- 计分奖励数值（line_base/combo/all-clear）与**均匀等概率**是建模假设；真实游戏可能用**自适应 RNG**
  （按棋盘发顺/坑你），会改变比例。
- 真正逼近 oracle 需要**学习型价值函数（RL）**，本项目止步于"搜索已收敛"的证据，未训练 RL。

---

## 复现

零依赖，纯 Python 标准库（含自写 SVG 绘图）。

```bash
python3 pieces.py            # 看 38 种方块目录
python3 sim.py 300           # 方差分解(可读版引擎)
python3 ladder.py 150        # 技能阶梯 + 交叉点
python3 compare.py 20        # 配对收敛(greedy基 vs strong基 rollout)
python3 experiments.py 200 24  # 跑全部 -> results.json
python3 plots.py             # results.json -> figures/*.svg
```

## 文件

| 文件 | 作用 |
|---|---|
| `pieces.py` | 38 种方块目录（旋转生成+去重） |
| `scoring.py` | 非线性计分模型（multi-clear/combo/all-clear，可调） |
| `sim.py` | 可读参考引擎 + 方差分解 |
| `fast.py` | bitboard 高速引擎 + beam-strong + lookahead + oracle rollout |
| `ladder.py` | 技能阶梯 + 交叉点 |
| `compare.py` | 配对头对头（CRN，验证 rollout 回归） |
| `experiments.py` | 一键产出 `results.json` |
| `plots.py` | 零依赖 SVG 出图 |

## License
MIT
