# Option-3 执行规格：Part A 可靠性修正 + 4×4 双 gate 认证 harness（8×8 训练之前）

> 三轮设计审核已 APPROVE-WITH-CHANGES（见 memory/rl_plan.md）。本规格是**具体实现级**的执行清单，
> 待新 agent 审查后执行。**不含 8×8 训练**（那是双 gate 通过后的下一阶段）。

## Part A — 修正已发布的封口结论（改现有代码/输出/README）

### A1. 通道A 存活 hazard 加采样 CI（oracle_analysis.py: survival_curve）
- 新增纯 Python helper `poisson_upper_95(k, exposure)`：k 死亡 / exposure 在险轮的**单侧** 95% 上界率
  = `0.5 * chi2_ppf(0.95, df=2(k+1)) / exposure`（单侧上界用 0.95 分位，**非** 0.975=双侧）。
  无 scipy → 查表常数 `CHI2_095_DF4 = 9.4877`（k=1）→ `0.5*9.4877/14365 = 3.30e-4`（已 python 验证）。
  ⚠️ 修正：原写的常数 2.3724 错误；第4轮审核误用双侧 0.975(→5.5716/3.88e-4)，本项目要单侧上界 → 用 4.7439 → **3.30e-4 / ≤1死每3028轮**。
- 另：README line 115 "贪心≈86%(缺口0.82[0.37,1.27])" 与 dp4.json(ratio=0.78, gap=0.615[-0.14,1.41]) **不一致**(陈旧)。
  A3 重跑 dp4 后，README 这两个数须同步为重跑值；competence gate 的"碾压 greedy"基线用**实测 ratio 0.78**(非0.86)。
- survival_curve 打印 + 写 survival.json：`seer_hazard_point`、`seer_hazard_poisson_ub95`、
  `seer_deaths`、`seer_at_risk`，并打印"≤1 死/{1/ub:.0f} 轮 (95%UB)"。
- 保留"seer≤真最优"为**独立建模界**，与采样 CI 分两行陈述。

### A2. EVPI 占比严谨化（oracle_analysis.py: channel_analysis + s_stability）
- (a) channel_analysis：对 per-seed 占比序列 `shares` 加 `bootstrap_ci(shares, _median, boot_seed=...)`，
  输出 `evpi_share_median_ci`。现仅有点中位数，无 CI。
- (b) 跑 3 个 T：`channel` 模式已支持 argv T 列表 → 用 `40,50,60`（验证第3个 T 的 cohort n≥20，否则降 T 或升 N）。
- (c) s_stability：**固定 cohort = 全 S 的 intersection-of-survivors**。现状各 S 用各自 `rb>=T` cohort
  (n=19/23/28/25) → 非 apples-to-apples。改：先对每个 S 跑 blind 得存活，cohort = {seed: seer 存活 ∧ ∀S blind(S) 存活}，
  再在此固定 cohort 上算各 S 的 EVPI_med。
- (d) README：通道B 段显式加一句"占比分母 (seer−strong) 载重于 strong=在线天花板假设；更强在线玩家会**抬升**占比
  （故现值对 strong 强度下偏保守）——方向1 RL 即测此假设"。保留 EVPI "≈"(D=3 截断下界)。

### A3. dp4.py 收敛 V* + 一致性
- docstring 第9行 "γ-折扣(γ=0.99)" → 改 "γ=0.95"（与 run 一致）。
- gate 用 `value_iteration(gamma=0.95, tol=1e-5)` 打印收敛 V*(empty)≈3.9158；保留现有 γ 敏感性。
- assert/报告：跑 **≥16k 种子**或报 **block-mean ± block-SE**（16 块×1000），不用单窗 3·SE
  （审核2 证伪了"系统偏差"：32k 种子缺口 −0.0096，单窗会被运气骗）。新增 `run_blocks(nblocks=16, per=1000)`。

## Part B — 4×4 双 gate 认证 harness（新文件 rl4.py + gate）

### 依赖（执行前置）
- **PyTorch（arm64 CPU/MPS）当前未装**（python3 无 torch/numpy）。afterstate 值网络需之。
  方案：`pip install torch`（arm64 wheel）。**破坏项目"零依赖"传统**——审查须裁决：装 torch / 还是 4×4
  用纯 Python 小 MLP（手写前向+反传）以保零依赖、但 8×8 仍需 torch。建议：装 torch，仅训练管线依赖，
  分析管线(oracle_analysis/dp4)保持零依赖。
- gate 的 ground truth 全是**纯 Python DP**（已有/新增），不依赖 torch → 即使 torch 缺位 DP 真值仍可算。

### B1. afterstate 值网络（4×4 版，certifies 8×8 pipeline）
- 4×4 单块/回合（dp4 简化①）→ afterstate = 放置后棋盘。state plane = 4×4 二进制。
- 两个 head/模式：
  - **mode-γ**：V(board)，FVI 目标 `mean_p max_pos(cl + γ V(b'))`，γ=0.95。
  - **mode-T**：V(board, k)，k=rounds-left，FVI 目标 `mean_p max_pos(cl + V(b', k−1))`，V(·,0)=0，**无折扣**。
- 小 MLP（16→64→64→1，ReLU；mode-T 把 k/T 归一化拼进输入）。warm-start 用 heuristic4(board) 预拟合一轮（见 rl_plan F2）。
- 目标变换：4×4 无 combo（dp4 简化②）→ 奖励有界(≤8)，**mode-T 在 4×4 不需 log1p**；但 harness 须**预留** target-transform 钩子
  （8×8 才启用，见 rl_plan F1），并在 4×4 验证"transform=identity 时 == 不变"。

### B2. 两个真值 DP（纯 Python，新增于 dp4.py）
- **mode-γ 真值**：已有 `value_iteration(γ=0.95)` → V*(empty)=3.9158。
- **mode-T 真值（新增 ~20 行）**：`backward_dp_T(T, reachable, moves)`：
  `Vk[b] = mean_p max(0, max_pos(cl + V_{k-1}[b']))`，V_0≡0，迭代 T 次（无折扣有限 horizon 后向归纳）。
  返回 `V_T[0]` = 从空盘 T 回合无折扣最优期望分。**这是 mode-T RL head 的认证靶**。

### B3. 双 gate 判据（预注册，两者皆过才认证 8×8 pipeline）
- **γ-gate**：|V_net(empty) − 3.9158| < ε_v(如 0.05)；且 greedy-on-V_net 最优比 ≥ 0.95（优于近视贪心 0.86）。
- **T-gate**：对 T∈{8,16}，|V_net(empty,T) − backward_dp_T(T)| / backward_dp_T(T) < 5%；
  且 V_net 诱导策略的在线无折扣均分 ≈ backward_dp_T(T)（paired CRN，effect size 预注册）。
- **位移检查**（rl_plan F2）：训练后 corr(V_net, heuristic4) 须显著 < 1（证明 V 离开了 init 盆地）。
- **已知不认证项**（须在 README/log 声明，类 R3 的 F3）：4×4 是**单块/回合**，故双 gate 认证
  {值网络 + FVI 循环 + γ与T两种 backup + 真值复现}，**不**认证 8×8 的"beam_hand 三块手牌候选枚举 × afterstate"集成
  —— 那一层的正确性靠 8×8 competence gate（碾压 greedy + 配对 CRN 平/胜 beam-strong）兜底。

## 执行顺序
A3(dp4 收敛 V*) → A1/A2(纯 Python，可立即跑) → 决策 torch → B2(纯 Python 真值 DP，可先写) → B1(torch 网络) → B3(双 gate)。
A 部分本会话可全部完成；B 部分 torch 装好后 4×4 训练分钟级（reachable 仅 41503）。
