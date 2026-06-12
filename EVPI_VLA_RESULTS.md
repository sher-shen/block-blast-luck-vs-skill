# 目标线② 收口 + 连击豪赌计算器（2026-06-12）

两件交付，均在私有 repo `block-blast-luck-vs-skill`，N=120 / D_oracle=3 口径，零新建模假设：

1. **verdict(i) 落地：用最强可玩在线策略 vla 当 EVPI 占比的新分母重算"运气=信息价值"占比。**
2. **连击豪赌计算器**：把超线性 combo 当 press-your-luck 赌局量化（解析 + strong 实测标定）。

---

## 1. EVPI 占比：换在线天花板 strong→vla → 运气占比抬升（预注册 verdict(i) 命中）

**背景**：原 channelB 把休闲玩家到 oracle(seer, 完美前瞻) 的缺口拆成
`raw(seer−strong) = EVPI(seer−blind) + procedure(blind−strong)`，运气占比 = EVPI/raw 的 per-seed
配对中位数 = **69/65/57%**（T=40/50/60）。ROADMAP 预注册风险(d)+verdict(i)：分母载重于
"strong=在线天花板"这个假设，**更强的在线玩家会抬升占比**（占比对在线强度单调上偏保守）。
交付的 vla（价值引导前瞻 D2 S30 + 训练 V，2924 ≫ strong 2364）正命中此前提。

**做法**（`evpi_vla.py` → `evpi_vla.json`，图 `figures/evpi_vla.svg`）：
- 三 oracle 玩家用 oracle_analysis 的**原函数**重跑（确定性、同种子）⇒ **必复现 channelB 的 cohort 与占比**。
- vla 喂**同一 `_deal_stream(seed)`** ⇒ 与 strong/blind/seer 严格 CRN 配对；T_MAX=50（网络 k 归一化常数）。
- cohort4 = {strong, blind, seer, **vla** 都活到 T} 的冻结 intersection-of-survivors；在同一 cohort4 上
  同时报 strong-分母与 vla-分母占比 = apples-to-apples 隔离"换分母"效应。

**自洽 sanity check（关键）**：strong-分母 cohort3 占比 = **69/65/57%**（n=54/41/34），与 channelB.json
**逐点吻合** ⇒ 重跑可信，vla 数字可信。

| T | cohort4 n | 占比(strong 分母, 同 cohort4) | **占比(vla 分母, verdict i)** | seer−blind(EVPI) | seer−vla 残差 | blind vs vla |
|---|---|---|---|---|---|---|
| 40 | 39 | 65% [48,75] | **109% [99,127]** | 835 | 754 | vla>blind (+81) |
| 50 | 26 | 63% [52,71] | **106% [87,137]** | 1133 | 1048 | vla>blind (+85) |
| 60 | 20 | 56% [51,77] | **96% [78,134]** | 1384 | 1426 | blind>vla (−42, 噪声内) |

分数阶梯(T=50, cohort4 中位数)：strong **2703** < blind **3250** < vla **3460** < seer **4395**。

**结论（verdict(i) 命中，诚实）**：把在线天花板从 strong 换成最强可玩策略 vla 后，运气(信息价值)占比从
**69/65/57% 升到 109/106/96%（≈100%）**。在同一 cohort4 上 strong-分母仅 65/63/56% ⇒ **抬升来自换分母本身，
不是 cohort 变化**。含义：**面对最强的"无真未来"可玩策略，到完美前瞻 oracle 的残差缺口几乎全是不可约运气
（信息价值）——可学的"过程/技能"通道已被 vla 吃满。**

**为什么占比可 >100%（不是 bug，是信号）**：vla 本身就是比 blind 更强的边缘化器（无真未来但 MC-rollout+学习 V），
在 T40/T50 上 vla>blind ⇒ procedure(blind−vla)<0 ⇒ seer−vla < seer−blind ⇒ 占比 >100%。超出 100% 的部分 =
vla 比朴素 blind 边缘化器多赚的量。占比 ≈100% 的正确读法 = "残差 ≈ 纯 EVPI"。

**诚实边界**：cohort4 比 cohort3 小（39/26/20，因要求 vla 也存活）⇒ CI 变宽、跨 100%，尤其 T60(n=20)；
T60 上 vla 略逊 blind（−42，噪声内）。占比的**方向性结论（单调上偏、随天花板变强而升）稳**；**点值 ≈100% 带宽 CI**。
未改 n=25 主 headline / 未改 channelB 原结论；这是预注册三选一里 (i) 的兑现。

---

## 2. 连击豪赌计算器（`combo_gamble.py` → `combo_gamble.json` / `_calc.json`，图 `figures/combo_gamble.svg`）

**动机**：combo 在**每次放置**触发消除时 +1、任一次不消则清零，奖励随层级**超线性**上涨 → 经典
press-your-luck 结构。**这条超线性正是本游戏高方差(运气重)的物理来源**，直接服务 luck/skill 主线。

**(A) 解析计算器**：把连击建模为"每步以维持概率 p 续命的几何赌局"，层级 s 一次消除给 combo 价值 u·(s−1)。
正在 C 层、之后每步维持 p 的期望未来 combo 奖励闭式
`G(C,p) = u·p/(1−p)·[(C−1)+1/(1−p)]`；岔路"保连击 vs 断连击"的纯 combo EV 增益
`ΔG(C,p)=[u·C+G(C+1,p)]−G(0,p)` = 你为保连击愿付的棋盘健康度代价上限；可反解 break-even p。

**(B) strong 实测标定**（用 `beam_hand_path` 复刻 strong 逐块落点 + 逐次放置重放 score_placement）：

| 量 | 实测值 | 含义 |
|---|---|---|
| p_maintain(整体) | **0.44** | strong 约 44% 的放置触发消除 |
| **combo 奖励占总分** | **53%** | 超过一半的分来自 combo 超线性 = 运气引擎 |
| L̄(每消条数) | 1.03 | strong 几乎都是单条消除，多条罕见 |
| 连击链 均长/最长 | 1.52 / 6 | 长连击罕见；87% 的链 ≤2 |
| p_maintain **按层级** | 0.52→0.40→0.26→0.20→0.16 | **维持概率随层级速降**（强消会越搞越乱棋盘） |

**核心诚实发现**：常数-p 几何模型**高估深连击赌局 2–5×**——因为真实维持概率随层级急降。
assumed 档 u=50 下，G_const(C=3)=150 vs **G_emp(衰减 p)=38（高估 4×）**。

**统一结论**：combo 占分高达 53%（运气引擎确凿），但价值来自**大量短链**而非英雄式长连击；
**刻意追逐深连击是亏本押注**（每多一层既要消除又恶化棋盘，下一步维持概率掉到 0.16）。
两条都真、互补：combo 在总量上主宰得分（→ 高方差 → 运气重），但深连击的边际赌局赔率差。

---

## 复现
```
python evpi_vla.py 120        # ~30min(oracle 6min CPU 并行 + vla 25min GPU) → evpi_vla.json
python plot_evpi_vla.py       # → figures/evpi_vla.svg
python combo_gamble.py empirical 120 50   # ~2s → combo_gamble.json
python plot_combo_gamble.py   # → figures/combo_gamble.svg
```
口径守则：N=120 D_oracle=3；CRN 同 `deal-{seed}`；vla T_MAX=50；占比 = per-seed 配对中位数 + bootstrap CI；
未触 n=25 主 headline，未改 channelB 原结论。
