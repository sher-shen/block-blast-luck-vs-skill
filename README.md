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
4. **【创新指标】信息价值 = 运气**（见 [`oracle_analysis.py`](oracle_analysis.py)）：
   让一个**先知(seer)** 在 rollout 时使用**真实未来方块**，对比一个**结构完全相同、却只能看到未来
   _分布_（采样未来取平均）的 blind 玩家**——两者唯一差别就是"知不知道这一局的真实未来"。其分差
   = **期望完美信息价值 (EVPI)** = 运气的量化。

> ⚠️ **重要修正（2026-05-31，经三轮独立审核）**：早期版本用
> `1 − strong均分/oracle均分 ≈ 78%` 当运气占比。后来发现把 oracle 升级成 beam-rollout + 真未来后，
> 它**近乎不死**——无界对局里能活几万轮、分数滚到 1e5–1e6。于是"分差"被**存活长度**主导、
> **本质无界**，那个 78% 的分母 ill-defined，**已撤回**（见 [`results.json`](results.json) `oracle_RETRACTED`）。
> 正确做法是把运气拆成**两条通道**，且都在**固定 horizon T**（所有玩家只跑 T 轮，分数才可比）下度量：
>
> - **通道 A · 存活运气**：各玩家活过 t 轮的比例。先知几乎不死 → "死"基本可避免=技能；
>   不可约的存活运气 = 罕见的 [Burgiel 杀手序列](https://www.semanticscholar.org/paper/How-to-lose-at-Tetris-Burgiel/11c12871bfa138fa8bb93a4e5dbcca36c5d214fa)，
>   以先知的**每轮死亡 hazard** 给出上界（真最优活得 ≥ 先知）。
> - **通道 B · 计分运气 (EVPI)** = `seer − blind`，**只在"strong/blind/seer 都活到 T"的 cohort 上算**
>   （否则死亡玩家分数被冻结，会把存活差异漏回计分）。再把先知的总优势带符号切分：
>   `raw(seer−strong) = EVPI(信息) + procedure(blind−strong, 搜索本身)`，**不 clip**。
>
> 三轮审核抓出并修正的关键陷阱：① blind 可能 < strong（*rollout 回归*）→ procedure 须带符号、地板用
> strong；② 死亡冻结分污染计分 → cohort 条件化；③ cohort 只留"strong 能活"的易种子 → 算出的
> 运气占比是**下界**（顶部难局的运气计入通道 A）；④ 用"自信猜错"的 anti 会高估信息价值 → headline
> 用 `seer−blind`；⑤ EVPI 两端皆有偏 → 只报 "≈"，靠 **EVPI 随采样数 S、前瞻深度 D 趋平** 背书。

---

## 结论

![players](figures/fig1_players.svg)
![ladder](figures/fig2_ladder.svg)

数据：38 种方块，方差/阶梯 200 种子（[`results.json`](results.json)）；
运气两通道 N=120 种子、D=3、固定 horizon（[`survival.json`](survival.json) / [`channelB.json`](channelB.json)）。

**① 技能地板巨大——这是"技能游戏"的一面。**
乱放均分 **57**，会玩（strong）**6453**，差 **≈113×**。光"不犯傻"就决定了两个数量级。

**② 熟练玩家之间，运气≈技能——这是"运气游戏"的一面。**
方差分解（greedy vs strong）：**技能 34% / 运气 34% / 交互 32%**。
而那 32% 交互项是"**只有会玩的人才兑得出的好牌机会**"，本质偏运气。

**③ 分界不是一刀切，而是在很高处。**
技能阶梯显示：分数 **~3400（交叉点）以下几乎全程技能主导**（缺口/σ 从 139 一路降到 ~6，都 >1），
只有逼近天花板才翻成运气主导。**所以"低分=没练好、不是运气差"对绝大多数水平都成立。**

**④ 运气分两通道（固定 horizon，N=120，D=3，经三轮审核）。**

![survival](figures/fig3_survival.svg)
![evpi](figures/fig4_evpi.svg)

- **通道 A · 存活运气 ≈ 极小。** 先知活过 t 的比例**全程 ≈ 1.00**（t=120 时 0.99），
  **每轮死亡 hazard ≤ 7×10⁻⁵/轮**（14365 轮里仅 1 死）= 杀手序列率上界。对照在线 strong 从
  0.91(t=20) 一路掉到 0.38(t=120)、blind 更差(0.20)。**含义：有前瞻则游戏几乎不可输 → 普通玩家的"死"
  绝大多数是可避免的技术问题，不是运气；不可约的存活运气只剩罕见的 Burgiel 必死牌。**
- **通道 B · 计分运气 (EVPI) ≈ 先知得分优势的 6 成。** 在"都活到 T"的 cohort 上：
  - T=40 (n=54)：`raw 1442 = EVPI 943 [816,1082] + procedure 499`，信息占比 **69%**。
  - T=60 (n=34)：`raw 2281 = EVPI 1427 [1221,1662] + procedure 854`，信息占比 **57%**。
  - 即先知相对最强在线玩家的得分优势里，**~57–69% 是纯粹"知道未来"的价值（运气）**，其余是前瞻搜索
    本身的功劳。该比例随采样数 S∈{4..32} 与前瞻深度 D（D≥3 即饱和）**稳定**。

**总结**：Block Blast 是**"技能定地板、运气定天花板"**的游戏，但天花板的"运气"有两副面孔——
**存活几乎全靠技能**（完美前瞻下近乎不死，真运气只剩罕见杀手序列），而**给定存活、想多刷分则约六成靠
牌运**（EVPI 占先知优势 57–69%）。中低水平拼策略（地板 **113×**），顶端在"刷分"维度拼牌运。它与五子棋
（运气=0、有必胜策略）不同范式：这里**无必胜策略**，连无限存活都被
[Burgiel《How to lose at Tetris》](https://www.semanticscholar.org/paper/How-to-lose-at-Tetris-Burgiel/11c12871bfa138fa8bb93a4e5dbcca36c5d214fa)
式杀手序列否定——只是那种序列**极罕见**。

---

## 诚实的局限

- **EVPI 是 "≈" 不是 "="**：seer 只前瞻 D 手（D≥3 已饱和）是真离线最优的**下界**；blind 用 S 份采样
  近似"按分布最优"，仍有蒙特卡洛噪声。两端偏差方向相反，故 EVPI 只报量级 + CI，靠"随 S/D 趋平"背书。
- **cohort 选择偏差 → 计分运气占比是下界**：通道 B 只在"strong 也能活到 T"的种子上算，而这些是**较易的局**；
  顶端难局（strong 早死）的运气被计入**通道 A**。所以"信息占比 57–69%"是计分运气的**保守下界**。
- **存活 hazard 是上界**：真最优活得 ≥ 我们的 seer，故 7×10⁻⁵/轮 是杀手序列率的**上界**（真值更小）。
- **结论依赖计分模型**：line_base/combo/all-clear 数值与**均匀等概率**发牌是建模假设；先知正是靠
  combo 复利刷分，真实游戏若用**自适应 RNG**（按棋盘坑你）会改变比例。计分敏感性分析为后续工作。
- 真正逼近"在线天花板"需**学习型价值函数（RL/DQN）**；本项目止步于"搜索已收敛 + 信息价值已量化"，未训练 RL。
- **4×4 精确 DP** 基准（量化启发式离真最优多远）作为可解类比，列为后续。

---

## 复现

零依赖，纯 Python 标准库（含自写 SVG 绘图）。

```bash
python3 pieces.py            # 看 38 种方块目录
python3 sim.py 300           # 方差分解(可读版引擎)
python3 ladder.py 150        # 技能阶梯 + 交叉点
python3 compare.py 20        # 配对收敛(greedy基 vs strong基 rollout)
python3 experiments.py 200 24  # 方差/阶梯/收敛 -> results.json
python3 plots.py             # results.json -> figures/fig1,fig2

# 运气两通道(创新主线, 经三轮审核; D=3 已由 D-sweep 定为 plateau)
python3 oracle_analysis.py sweep 24 80      # D-sweep: 定前瞻深度 D(存活/分数 plateau)
python3 oracle_analysis.py sstab 40 3 40    # EVPI 随采样数 S 是否趋平
python3 oracle_analysis.py survival 120 3   # 通道A 存活曲线 -> survival.json
python3 oracle_analysis.py channel 120 3 40,60  # 通道B EVPI 分解 -> channelB.json
python3 plots_oracle.py      # survival/channelB.json -> figures/fig3,fig4
```

## 文件

| 文件 | 作用 |
|---|---|
| `pieces.py` | 38 种方块目录（旋转生成+去重） |
| `scoring.py` | 非线性计分模型（multi-clear/combo/all-clear，可调） |
| `sim.py` | 可读参考引擎 + 方差分解 |
| `fast.py` | bitboard 高速引擎 + beam-strong + lookahead + beam rollout |
| `ladder.py` | 技能阶梯 + 交叉点 |
| `compare.py` | 配对头对头（CRN，验证 rollout 回归） |
| `experiments.py` | 方差/阶梯/收敛 -> `results.json` |
| **`oracle_analysis.py`** | **运气两通道（创新主线）：seer/blind/anti 玩家 + 固定 horizon + 存活曲线 + EVPI 分解 + bootstrap CI** |
| `plots.py` | 零依赖 SVG（fig1 玩家 / fig2 阶梯） |
| `plots_oracle.py` | 零依赖 SVG（fig3 存活 / fig4 EVPI 分解） |

## License
MIT
