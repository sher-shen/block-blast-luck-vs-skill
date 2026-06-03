# 方向1 RL 方案 — afterstate 值迭代当"在线天花板压力测试"（2026-05-31 三轮审核 APPROVE-WITH-CHANGES）

## 定位（关键，别退回 solver 框架）
RL **不是**"造更强的 bot"——那是项目刻意拒绝的 solver 框架，也是"弱 agent 陷阱"所在
（beam-strong 已近启发式族天花板，续2 已证 LA−strong 统计打平）。
RL 的科学价值 = **收敛三脚架的第三条腿**：EVPI 信息占比 57–69% 的载重假设是
"strong(beam,无前瞻) = 真在线天花板"，此前只用"加搜索深度无用"测过一次；
用一个**学习型值函数**（与搜索正交的逼近器）独立再测一次：
- RL 打平 strong → 天花板两正交方向三角验证，headline **加强**；
- RL 胜 strong → 天花板被低估，EVPI 占比是**高估**，须重算（占比会**升**，见下）；
- RL 输 strong **但通过 4×4 gate** → "预算内未达搜索天花板，inconclusive"，**永不当结论**。

## 方法
afterstate 拟合值迭代（FVI），**非 DQN/PPO**。理由：放置动力学确定、每手即时分可观测、
直接复用 beam_hand 枚举候选 → 学 V(afterstate) 比在巨大可变动作空间学 Q(s,a) 高效得多。
- **afterstate = 末手棋盘**（3 块全放完、combo 已贯穿）；与 beam_hand 返回的 (board,combo,score) 同粒度（已验）。
- 动作目标：argmax over beam 候选的 `hand_score + V(afterstate)`，**目标里不含 heuristic**（V 已编码未来，加 heuristic=重复计数）。
- 但 **V 用 heuristic(board) warm-start 初始化**（避免冷启停滞）；heuristic 只进 init，不进 bellman target。
- 状态特征：8×8 棋盘面 + 3 手牌 glyph 面 + combo(分桶并 cap，cap 须覆盖 seer 触及的 combo 域，报越界比例) + rounds-remaining。
- 网络：小 CNN（2–3 conv 3×3 / 32–64ch → concat → MLP → 标量 V），~50–200k 参数；Mac CPU/MPS 均可。

## 两套折扣制度（务必分开——审核2/3 抓的核心 apples-to-oranges）
- **4×4 dp4 gate：γ=0.95 折扣**（因 dp4.py 的 V* 本身 γ 折扣）。同一 pipeline 须复现 dp4 收敛 V*≈**3.9157**
  （VI tol≤1e-5，**不是** tol=1e-3 的 3.91）；gate 跑 **≥16k 种子**（或报 block-mean±block-SE，单窗会被运气骗，见下）；
  RL-greedy-on-V 须 ≥ 近视贪心的 86%-of-optimal，理想 →~100%。
- **8×8 competence gate + CRN eval：无折扣有限 horizon-T 值**（V 条件于 rounds-remaining k），匹配 channelB 的固定-T 无折扣求和。
  **绝不**把 γ=0.95 策略塞进固定-T 的 CRN 比较（γ=0.95 有效 horizon~20 手，会低配 channelB 全额奖励的存活尾部 → 可能"为错误的理由"输）。

## 审核三轮抓出、必须落地的改动
1. **(F1) combo 奖励无界** → 无折扣有限-T 的 V 目标在存活密集态可达 1e4–1e5 重尾，MSE 会被极端态主导。
   目标须 **log1p / Huber 变换**（或预测 per-round-rate×k），报尾部/裁剪比例。否则 8×8 训练大概率不稳。
2. **(F3 最锋利) 4×4 单 γ-gate 只认证"机器收敛到折扣 V*"，不认证无折扣-T k-条件目标。**
   须加**第二个 4×4 微 gate**：4×4 引擎跑无折扣有限-T + k-条件，对照 4×4 reachable 集上的**后向有限-horizon DP**（dp4.py 已有 forward-DP 料，后向~20 行）。
   只有 {γ-gate} ∪ {无折扣-T gate} 才完整认证 8×8 pipeline。
3. **(F2) warm-start vs 不进 target**：FVI 会离开 heuristic 盆地（target 只含 hand_score+V），但须**预注册"V 已离开 init"位移检查**
   （V_final 与 heuristic 相关性跌破阈值 / 探针集值变化 > greedy-vs-strong 缺口）；否则"RL≈strong"可能是 V 没动的假象。
4. **(F4) 预算 + 防自欺**：wall-clock 预算须含 k-条件的 ~T× 样本膨胀 + 每候选每手 beam 枚举成本；
   预注册学习曲线 plateau 判据（冻结探针集值末段 X% 持平），否则 B5(iii) 的"inconclusive"逃生门会被选择性援引。
5. competence gate(b)：vs beam-strong 须 **paired-CRN + 预注册效应量**（非仅符号——续2 LA−strong=−1214±1814 是统计打平，别自欺），eval 种子**冻结且与训练不相交**。

## 预注册结果表（训练前写死）
- (i) RL≫strong：天花板低估 → **在同一冻结 intersection-of-survivors cohort 上**重算 EVPI 占比（用 RL 当新分母/天花板，**不**用 RL 自己的存活集，否则重引 cohort 选择偏差）；占比只会**升**。
- (ii) RL≈strong（且 4×4 双 gate 过 + greedy 被碾压 + V 已离开 init）：天花板三角验证，headline 加强。
- (iii) RL≪strong 但过 4×4 gate：inconclusive，不当结论。

## 顺序（reviews 隐含的最优执行链）
(a) 修 dp4 打印收敛 V*（tol≤1e-5）+ Part A 可靠性修正；(b) 建 4×4 **双 gate** 认证 harness（γ-折扣 + 无折扣-T 后向 DP）= 低风险、有界、先 de-risk 一切的"第一块可执行"；(c) 才上 8×8 训练。

见 [[oracle_immortality_reframe]] 六大陷阱 + [[strategy_design]] 收敛论证 + log.md 续6。
