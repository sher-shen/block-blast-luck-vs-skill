# oracle 近乎不死 → "运气=信息价值"必须重设计（2026-05-31 重大发现）

## 发现
把 oracle 从"D=3 greedy 偷看未来"升级成 **beam-rollout + 看真实未来(seer)** 后，
它在 8×8 上 **近乎不死**：无界 horizon 下能活几万轮，分数爆炸到 **1e5–1e6**
（每步都消除→combo +50 连击复利→超线性）。抽查 seed3 seer=2.6M vs strong=2081。

## 为什么这推翻了旧 headline
旧"运气 = oracle 缺口 = `1 − strong_mean/seer_mean`"是 **ratio-of-means**，而 seer 分数
被【存活长度】主导、本质无界 → 这个比值口径**坏掉**（不只是长尾偏高，是分母 ill-defined）。
已发布的 **78% 必须撤回**，不是收紧 CI 能救的。

## 正确口径：固定 horizon T + 两条运气通道
所有玩家只跑 **T 轮**（分数才可比、有限）。运气拆成：
- **通道 A 存活运气** = Burgiel 杀手序列。报 seer 的【每轮死亡 hazard】(=死亡数/总在险轮数)，
  是真最优 hazard 的**上界**(真最优活得≥seer)。结论是整条 survival 曲线，不是单个 T 的点。
  含义：有前瞻时几乎不可输 → 休闲玩家的"死"大多可避免=技能；不可约运气=罕见杀手序列。
- **通道 B 计分运气 = EVPI**(期望完美信息价值) = `seer − blind`。
  - blind = 前瞻但 rollout 用 **采样的随机未来取均值**(无真信息的 Bayes 边缘化臂)。
  - 必须在【strong/blind/seer 都活到 T 的 cohort】上算：否则死亡玩家分数被冻结，
    把存活差异漏回计分通道（审核2 抓的关键 bug）。
  - 带符号切分：`raw(seer−strong) = EVPI(seer−blind) + procedure(blind−strong)`，
    **不 clip**（procedure 可为负=rollout 回归，clip 会掩盖它）。
  - 份额 = **per-seed 配对中位数** `(seer_i−blind_i)/(seer_i−strong_i)`，不是 ratio-of-means。

## 关键陷阱（审核三轮抓出，复用时务必记住）
1. **blind 可能 < strong**（rollout 回归）→ 份额可 >100%。所以地板=strong，procedure 带符号报。
2. **死亡冻结分** → 固定 T 总分混入存活。必须 cohort 条件化或 per-round。
3. **cohort 选择偏差**：只留 strong 能活的=易种子，EVPI 在易种子偏低 → cohort 份额是运气**下界**
   （杀手序列运气在被排除的种子里，归通道 A）。写作时必须声明方向。
4. **anti(单条假未来当真)** 是"自信地猜错"，分数畸低 → `seer−anti` 会**高估**信息价值；
   只能当 single-draw-bias 诊断，headline 用 `seer−blind`。
5. EVPI 两端都有偏(seer 是 D 截断下界 / blind 是 S 噪声边缘化) → headline 只能说 "≈"，
   靠 **s_stability(EVPI 随 S 趋平)** + **d_sweep(随 D 趋平)** 背书。
6. `raw=EVPI+procedure` 是**代数恒等**，别吹成"发现"，报"缺口如何切分"。

## 最终落地数字（N=120, D=3, 已 push commit 900d63b）
- **通道A 存活**：seer 死亡 hazard **点估计 7e-5/轮**(14365轮1死；单侧 95% Poisson 上界 **3.3e-4/轮**——勿写"≤7e-5"，那是点估计非界)；strong 存活 .91→.38(t20→120)，blind 更差(.20)。
- **通道B EVPI**：T=40(cohort54) EVPI **943 [816,1082]**, raw1442, 信息占比 **69%**；
  T=60(cohort34) EVPI **1427 [1221,1662]**, raw2281, 占比 **57%** → 报范围 **57–69%**(随 T 变，非定值)。
- S-stability：EVPI 随 S∈{4,8,16,32}=1080/930/940/1060，~±10% 噪声内平 → S=8 够。
- D-sweep：seer 存活 D=3 即饱和 1.00，分数 D=3≈D=5(7851/7844) → D=3 plateau。
- 已撤回 results.json 的 oracle_gap_ratio(留 oracle_RETRACTED 说明)；新数据 survival.json/channelB.json/sstab.json。

## 已发布结论的可靠性修正（续6 三轮审核 Part A 抓出）—— ✅ 已全部落地（commit `7d90211`，2026-05-31）
> 落地证据：hazard Poisson 上界已进 survival.json/README；占比 CI + 第3个T(50) 已进 channelB.json
> （57–69% 现为 T=40/50/60 三点 + bootstrap CI）；sstab 已改固定 intersection cohort（sstab.json:
> fixed_cohort_n=24，EVPI_med 1085/980/1055/1065 仍平 → S-稳定真实成立）；dp4 γ docstring 不一致已在
> rl8 Phase 0 修（0.99→0.95）。下文保留作"当时抓出什么"的记录：
- **通道A hazard 只基于 1 次死亡**（seed94 @round85，1死/14365在险轮）。"≤7e-5/轮"是**点估计**不是界。
  须并报 **Poisson 单侧 95% 上界 = 3.30e-4/轮（≤1死/~3028轮）**；"seer≤真最优"是另一条**建模**界，与采样 CI 分开陈述。定性结论(有前瞻几乎不死)稳，只是数字要 CI。
- **EVPI 占比**：(a) 须报**占比比值本身的 bootstrap CI**（现仅 EVPI level 有 CI）；(b) 加第3个 T(如50/80)，"57–69%"现是 2 点线段；
  (c) **sstab 的 S-平坦是假象**：各 S 用了**不同 cohort**(n=19/23/28/25)→非 apples-to-apples，须固定为全 S 的 intersection-of-survivors 再比；
  (d) 占比分母(seer−strong)载重于"strong=天花板"，**更强在线玩家会抬升占比**(占比对 strong 强度是下偏保守)——正是 RL 要测的。
- **dp4 online>V* 不是系统偏差**(审核2 跑 32k 种子证伪：缺口−0.0096，4升4降)。Round1 的"+1.69SE"是单种子窗运气。
  但 gate 须跑 ≥16k 种子 + 打印收敛 V*(tol≤1e-5)，别用单窗的 3.91。dp4 docstring 写 γ=0.99 而 run 用 0.95，内部不一致须修。

## 工程
- Python print 到管道**默认缓冲**，后台跑看不到进度 → 所有实验 print 加 `flush=True`。
- **stale `__pycache__` 幻象不可复现**(续7 踩)：编辑 .py 后若某次调用加载到旧 .pyc，同一确定性函数会给出
  不同结果(dp4 greedy 一度三值 0.78/0.86/5.26)。诊断"数值非确定/不可复现"前**先 `rm -rf __pycache__`**；
  committed JSON 与现码输出不符时，优先怀疑**产物陈旧**(用旧码生成、未随码重生)而非代码 bug。
- **后台运行别叠加**：`nohup … &` + 工具 run_in_background = 双重后台，状态/退出码混乱。只用其一。
- **跨 N/D 不可比**：oracle_analysis channel/sstab 默认 D=5，但 baseline 是 **N=120 D=3**；重跑务必显式传 `120 3`，否则 EVPI/份额变化是 D/N 假象不是真结果。
- 见 [[strategy_design]] 路线图、[[project_goal]] 创新主线、[[rl_plan]] 方向1 设计、log.md 续4/续6。
