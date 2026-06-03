# ROADMAP — Block Blast 8×8

> 高层"目标对齐"视图。**实现级细节看** `PHASE_EXEC_PLAN.md`（① 4×4 双gate 方案 + ② 长期提纲）、
> `NEXT_PHASE_PLAN.md`（工作约定）、`PLAN_8x8_RL.md`（②a 的 /loop 可执行计划）。
> 本文件只答："做到哪了 / 接下来两条路各是什么 / 先做哪条"。配套 `GOAL.md`（目标定位）、`memory/MEMORY.md`。

## 当前状态快照（2026-06-01）

- [x] **正问题主线**：运气 = 后见之明价值（EVPI）双通道分解，N=120 / D=3 / 固定 horizon
- [x] **三条独立证据**：方差分解（技能34/运气34/交互32）、技能阶梯（交叉点≈3408）、搜索收敛
- [x] **通道 A 存活运气**：hazard 点估计 7×10⁻⁵/轮 + 单侧 95% Poisson 上界 3.3×10⁻⁴/轮
- [x] **通道 B 计分运气 (EVPI)**：57–69%（T=40/50/60，per-seed paired bootstrap 95% CI）
- [x] **4×4 精确 DP 锚点**：V*=3.9157；启发式近视≈在线最优 86%；discounted VOI≈在线最优 3×
- [x] **方向① 4×4 afterstate-FVI 双 gate**：**全 PASS**（γ-值/γ-策略/T-值/T-在线/位移），认证 FVI pipeline 可信，放行 8×8。详见 `memory/rl4_gate_results.md`
- [x] **可靠性修正已落地**：撤回 ill-defined 的"78%"；全数值带 CI、避免 ratio-of-means
- [x] **私有 repo 已推**：github.com/sher-shen/block-blast-luck-vs-skill
- [ ] **②a 8×8 RL** —— **下一步主攻**（4×4 gate 已为其铺路；计划见 `PLAN_8x8_RL.md`）
- [ ] **②b DDA 逆问题** —— 提纲已立，待 ②a 后

---

## 路线 ②a · 8×8 RL —— 最强可玩策略 + 在线天花板第三条证据腿【先做】

> 实现级见 `PLAN_8x8_RL.md`（/loop source-of-truth）+ `PHASE_EXEC_PLAN.md §②(a)` + `memory/rl_plan.md`。

- **双重目标**：(1) 实用——学出接近最优、不靠预知未来的 8×8 可玩策略（用户要的"怎么打更高分"）；(2) 认知——用学习型值函数（与 beam 搜索正交）独立再测"strong=真在线天花板"这一 EVPI 占比的载重假设。
- **关键子步骤**：afterstate-V 上 8×8（**无折扣有限-horizon-T**，条件于 rounds-left k，匹配 channelB）→ 小 CNN（棋盘面 + 3 手牌 glyph + combo 分桶 + k）→ 插入 CRN 框架（新 `play_rl` 替 strong 位）→ vs beam-strong paired-CRN 收敛测试。
- **风险**：(1) combo 奖励无界 → mode-T 目标须 log1p/Huber；(2) "RL≈strong"可能是 V 没动假象 → 预注册位移检查；(3) cohort 选择偏差 → 重算 EVPI 须用同一冻结 intersection-of-survivors cohort；(4) γ vs 固定-T apples-to-oranges；(5) 弱 agent 陷阱。
- **算力（单 Mac M-series CPU 量级估计）**：8×8 reachable 不可枚举 → rollout/self-play 采样训练 + k-条件 ~T× 膨胀 + 每候选 beam 枚举。预计单次训练**数小时~一两天**；CRN eval 另算数小时。**预注册 wall-clock 上限 + plateau 判据是硬约束。**
- **预注册结果表**（训练前写死，三选一，永不事后挑）：(i) RL≫strong→天花板低估，同一冻结 cohort 上用 RL 当新分母重算 EVPI（占比只升）；(ii) RL≈strong（且 4×4 gate 过 + greedy 被碾压 + V 已离开 init）→ 三角验证，headline 加强；(iii) RL≪strong 但过 4×4 gate → inconclusive，**永不当结论**。
- **工作量**：中—大。

## 路线 ②b · DDA 逆问题 —— 胜率可设计性 / 难度调控【待 ②a 后】

> 实现级见 `PHASE_EXEC_PLAN.md §②(b)` + `memory/project_goal.md` 长期方向节。

- **目标**：正问题（均匀发牌→运气/技能几成）已做；逆过来给定目标胜率 W*，设计发牌概率 θ（38 块非均匀）+ 计分参数命中它（= 游戏公司 DDA）。**卖点**：luck/skill 分解正是审计"难度旋钮诚实(改技能)还是操纵(改存活运气)"的工具。
- **关键子步骤**：forward `W(θ,π)` 用 strong/seer 当技能档 → 灵敏度 `∂W/∂θ_i`（**须 CRN 共种子有限差分**）→ 单调性 → 优化器（grid / 贝叶斯 / REINFORCE）。
- **风险**：θ 非均匀**同时**改 survival luck 和 skill ceiling（耦合）→ 须在固定 π 档分别报；DDA 文献成熟，需靠 luck/skill 归因差异化。
- **算力**：比 ②a 轻（纯模拟、无 GPU、无网络收敛风险），小时级~一天。**重跑必须显式传 `120 3`**（channel/sstab 默认 D=5）。
- **工作量**：中。

---

## 优先级：②a 先（已与用户拍板，2026-06-01）

理由：(1) ②b 全部卖点建立在 headline 站稳之上，逻辑上 ②a 是 ②b 前置；(2) 4×4 双 gate 刚 PASS，接 8×8 边际成本最低、风险最可控（有 inconclusive 逃生门）；(3) ②a 同时是用户真实想要的"最强可玩策略"，且"RL 当 luck 分母上界"**据我们检索未见直接先例**、竞争度低（此为内部优先级论证，非对外 headline；对外材料不写"首例"类绝对宣称）。
**建议链：②a 训 RL → 锁定 headline + 拿到强策略 → 再用稳固工具做 ②b 审计。**

## 依赖 / deadline
- ②a 依赖 4×4 双 gate（**已 PASS**）→ 前置已满足，可直接开。
- ②b 无硬前置，与 ②a 独立；单人单机建议串行。
- **无硬 deadline，兴趣驱动**。若锁定投稿窗口，②a 升级为"投稿前必做"，届时倒排。
- **硬性工作约定**：print 带 `flush=True`；数值带 CI 避免 ratio-of-means；诊断不可复现前先 `rm -rf __pycache__`；后台运行别叠加 `nohup &`+`run_in_background`。
