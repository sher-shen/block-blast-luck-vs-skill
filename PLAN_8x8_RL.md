# PLAN_8x8_RL — ②a 8×8 afterstate-RL 执行计划（/loop source-of-truth）

> **怎么用**：新开对话 `/loop /<skill 或直接贴本文件路径>`，loop 每轮读本文件 → 找到"当前阶段"→ 执行该阶段 → 更新状态块 → 命中 USER_GATE 就停下问我。
> **定位**：目标线①（最强可玩 8×8 策略）+ 目标线②（独立验证 "strong=在线天花板"）。见 `GOAL.md` / `ROADMAP.md`。
> **前置已满足**：4×4 afterstate-FVI 双 gate 全 PASS（`rl4_gate.json`），FVI pipeline 已认证。
> **repo**：私有 github.com/sher-shen/block-blast-luck-vs-skill。**commit/push 永远先问用户（USER_GATE）。**

---

## 0. 不可违反的硬约定（每轮开头自检）

1. **零依赖隔离**：torch 只在新文件 `rl8.py`（和现有 `rl4.py`）；`oracle_analysis.py / dp4.py / sim.py / fast.py / experiments.py` **绝不 import torch**。跑 torch 用 `.venv/bin/python`。
2. **apples-to-oranges 红线**：8×8 一律 **无折扣有限-horizon-T**（V 条件于 rounds-left k），匹配 channelB 的固定-T 无折扣求和。**绝不**把 γ=0.95 策略塞进固定-T 的 CRN 比较。
3. **数值带 CI、禁 ratio-of-means**：所有头对头报 paired bootstrap CI；EVPI 份额用 per-seed 配对中位数。
4. **诊断不可复现前先 `rm -rf __pycache__`**；committed JSON 与现码不符优先怀疑产物陈旧。
5. **print 全带 `flush=True`**；后台运行 `run_in_background` XOR `nohup &`，**绝不叠加**。
6. **cohort 红线**：重算 EVPI 占比必须用**同一冻结 intersection-of-survivors cohort**，**不**用 RL 自己的存活集（否则重引 cohort 选择偏差）。
7. **eval 种子冻结、且与训练种子不相交**。
8. **弱 agent 陷阱**：RL 打平 strong 是好结果不是失败；任何"训不过 strong"在预算/plateau 未达时一律判 inconclusive，**永不当结论**。
9. **预注册优先**：判据/效应量/wall-clock/plateau 在跑之前写死进本文件的状态块，事后不许改判据。
10. **channel/sstab 默认 D=5，baseline 是 N=120 D=3** → 重跑必须显式传 `120 3`。

---

## 1. 状态块（loop 每轮读这里，做完一阶段就更新）

```
CURRENT_PHASE: 1
PHASE_0_PREFLIGHT:   DONE   # 2026-06-01: __pycache__ 清; dp4 default gamma 0.99→0.95(V[0]=3.9157 复现); 移除单窗 assert 输出字段; README 补 3 引用(2402.12874/FUN2024/2306.14626)+空白点措辞降级; torch2.12.0 OK; 分析管线 0 torch
PHASE_1_INFRA:       IN_PROGRESS  # 2026-06-02: §3.0 迁移 gate PASS(T8 1.04%/T16 2.74%)。预算可行(~18h≪7天)。§3.6 定盘规则按用户审计**重写**(§3.6★ (a)-(d), 高-k 自洽真裁判)。instrument 落地 rl8.py(mc_rollout_value/collect_probes/selfcons_highk/offpolicy_states/board_planes_fast8 + CLI train8/check8)+smoke 全绿(放大签名检测器 work)。**logv T=50 训练进行中**(harness bg, 3h chunk, offpolicy 补采)。待: plateau→check8 定盘 logv 或测 rate/lograte 或 §3.6 USER_GATE → 才标 DONE
PHASE_2_TRAIN:       NOT_STARTED
PHASE_3_COMPETENCE:  NOT_STARTED
PHASE_4_VERDICT:     NOT_STARTED
PHASE_5_WRITEBACK:   NOT_STARTED
LAST_USER_GATE: 2026-06-02 §3.6 定盘规则**重写**（用户上一轮审计）：旧"训 T=3 比锚选最小"作废（区分不出 logv 无×k vs rate/lograte 被×k，T=3 horizon 太短）→ 改 (a)机理优先 logv (b)T=3 锚只证伪 bug (c)真裁判=高-k 自洽 V_net vs MC-rollout@k50 (d)全败→USER_GATE。已写进 §3.6★ + 状态块预注册，按 (a)-(d) 执行中。
NEXT_ACTION (新对话 /loop 从这里接 — ★STRENGTH-FIRST，见 §3.6★★ 用户治理覆盖):
  **TRANSFORM = rate**（补18 用户 §3.6(d) USER_GATE 定）：logv T=50 训练退化（expm1×max-自举顶 clamp、v8≈v16，补17/18）→ 弃；strength-first 用实战得分判强、不看 V 校准 → rate 线性稳收敛是正解，×k 偏置记目标②局限。稳定化：decode clamp(5000)+grad_clip1.0+lr5e-4（全在码里）。**"最强"真裁判 = Phase 3 实战比分**。
  **当前在跑**：rate 45-sweep 稳定性 probe（确认 rate 不退化：v8<v16、量级合理、不顶 cap）。
  **probe 过 → loop 接力做**：
  ① `python rl8.py train8 rate 50` 跑全程到 plateau（已配 grad_clip/lr/clamp；3h chunk，断点续 /tmp/rl8_8x8_rate_T50.pt；没 plateau 重复续）；
  ② plateau → **建 competence_gate**（新函数，复用 mc_rollout_value 骨架：rl-greedy-on-rate-V vs strong，冻结 EVAL_SEEDS、同种子 CRN 配对、固定 T=50、报 paired 得分差 + TOST + 16-block-SE；§5/§6）= **最强策略 verdict**；
  ③ 顺手 `python rl8.py check8 rate 50 2000 32` 跑**一次**高-k 自洽当 sanity（advisory，量化 ×k 偏置写进目标②局限，不阻塞）；
  ④ HARD_CEILING 终判 → PHASE_1_INFRA=DONE → 按 verdict 路由 Phase 4（rl≫strong=占比上修 / rl≈strong=佐证 / rl≪strong 但过 gate=inconclusive 局限）。
  **probe 仍退化（rate 也顶 cap/不分 horizon）→ 再 USER_GATE**：降 T 或 cap combo（用户上轮已排除，留作 fallback）。
  注意：Phase 3 比分被 beam 候选池同源限制（§5 正交性诊断：部分手枚举 full-action 确认 beam 没系统裁掉 rl 偏好落点）。
  关键文件：rl8.py(decode_vtot:288 / mc_rollout_value / selfcons_highk / offpolicy_states / net_forward_chunked / **待加 competence_gate**)、memory/rl8_phase1.md、log.md 续12 补1~18。MPS 默认。
PREREGISTERED (训练前填死，填后不许改):
  - TRANSFER_GATE_4x4:     rl8.py 的 8×8 CNN + 采样 FVI（非 rl4.py 的 MLP+全枚举）在 4×4 引擎复现 bdp_T(8)=3.5934/(16)=4.8637，rel<5% 才放行 8×8（见 §3.0，致命前置）
  - WALL_CLOCK_CAP_TRAIN:  (待 Phase 1 实测单 sweep 后**外推全程**填；复审警告真实成本可能一周+，不是"数小时")。**预注册硬上限 HARD_CEILING = 7 天**：Phase 1 末用「单 sweep × min_sweeps(≥T_MAX) × 安全系数 1.5」外推总训练 wall-clock；若 **外推 > HARD_CEILING → 直接 USER_GATE，不进 Phase 2**，报我重设计（降 T / 降采样态数 / 换 mode-γ 近似），**不是**跑到一半撞墙。训练中撞软上限(=外推值或人工设的中途上限)仍未 plateau = inconclusive→USER_GATE
  - PLATEAU_RULE:          冻结探针集(empty + 32 个固定采样 board × k∈{10,..,T}) 值末段连续 K=8 sweep 相对变化 < 0.5%
  - COMPETENCE_TEST:       **TOST 等价检验**（非"CI 含 0"——不显著≠相等）。预注册 margin **Δ_eq = 0.05 × strong 在 T=50 的总分均值**（单位写死 = **总分**，非逐轮率；Phase 1 用 strong eval 实测总分均值填死具体数值，填后不许改）；判 "≈" = rl−strong 的 90% CI 完全落在 [−Δ_eq,+Δ_eq]；判 "≫" = paired CI 下界 > +Δ_eq；判 "≪" = CI 上界 < −Δ_eq。M 须使 margin 内功效 ≥0.8（Phase 1 用 strong 方差估 M，复审：续2 σ_diff≈1800 极大，M 可能需远大于 2000）
  - EVAL_SEEDS:            M = (待 Phase 1 按功效定，下限由 TOST 功效算出) 冻结、与训练不相交。**fallback（若反解 M 超可承受上限，如 >5万 seed × 每 seed beam 成本）**：优先级 = 先降功效目标 0.8→0.7 报我，仍不够再放宽 Δ_eq（但放宽须连同 headline 措辞一起降级、写进局限），**绝不**事后缩 M 凑显著
  - HORIZON_T:             与 channelB 对齐用 {40,50,60}；competence 主报 T=50
  - TRANSFORM:             **= rate（per-round-rate×k）**（补18 用户 §3.6(d) USER_GATE 定）。logv 虽无 ×k 偏置但 **T=50 训练退化**（expm1×max-自举顶 clamp、v8≈v16 不分 horizon，补17/18）→ 弃。strength-first 用实战得分判强、不看 V 校准 → rate 线性稳收敛(4×4 已证)是正解；**×k 偏置记为目标②天花板局限**（高-k 自洽 sanity 时量化报）。稳定化：decode clamp(5000)+grad_clip1.0+lr5e-4。
  - SELFCONS_GATE_HIGHK:   §3.6★(c) 高-k 自洽**真裁判**（预注册，写后不许改）：探针 = empty + 32 个冻结中局态（strong+seer off-policy 采）@ k∈{25,50}；每探针 M_sc=2000 序列、报 paired-CI（禁 ratio-of-means）；PASS = 聚合中位 rel@k50<0.10 **且** rel@k50/rel@k25<2.0（无-×k 放大签名）。全候选失败 → §3.6 USER_GATE。
  - BUFFER_COVERAGE_τ:     eval 态（含 strong/seer 轨迹态）落在训练 buffer 支撑外的比例 < (待 Phase 1 填) 才放行（防 on-policy 分布漂移）
  - DISPLACEMENT_τ:        (待 Phase 1 实测 warm-init 与训练后探针集 RMS 差后填)
```

---

## 2. Phase 0 — Pre-flight 清理与小修（秒级，无训练）

目的：开训前清掉审查发现的 loose ends + 验证环境。**全部是非训练小修，跑完即可进 Phase 1。**

- [ ] `cd ~/Desktop/block-blast-sim && rm -rf __pycache__`
- [ ] **dp4.py 修两处内部不一致**（审查 B 维度发现，不影响 headline 但要清）：
  - `value_iteration` 默认签名 `gamma=0.99` → 改 `0.95`（与 docstring/README/所有调用一致）；改后跑 `value_iteration(tol=1e-5)` 确认仍 = 3.9157。
  - `run()` 主路径的单窗 `assert abs(mean_on−V[0]) < 3*se_on` → 移除或改走 `run_blocks`（tol=1e-5 + 16×1000 block-SE，gap=−0.0009 已证无系统偏差）；别让单窗噪声值出现在输出字段。
- [ ] **README 补 3 个对标引用**（审查 A 维度，related work 缺口）：(1) [arXiv:2402.12874](https://arxiv.org/abs/2402.12874) Skill or Luck? Return Decomposition via Advantage Functions（撇清与 advantage-based 单智能体分解的机制差别）；(2) [Gehnen & Venier, FUN 2024, Tetris Is Not Competitive](https://drops.dagstuhl.de/entities/document/10.4230/LIPIcs.FUN.2024.16)（通道 A 支柱，比 Burgiel 更贴）；(3) [arXiv:2306.14626](https://arxiv.org/abs/2306.14626)（**用 RL agent 估计关卡难度/完成率**——AUDIT-4 校正：更贴 ②b DDA，related work 双挂或措辞改"agent 表现作人类难度/完成率代理的先例"，**勿当 ②a 强策略先例**）。措辞避免"首次用信息价值定义游戏运气"，改"首次将 EVPI/survival 双通道分解用于单人随机生存 block-puzzle 并用精确 DP 锚定"。
- [ ] 验证 `.venv/bin/python -c "import torch; print(torch.__version__)"` OK；grep 确认分析管线 0 个 `import torch`。
- [ ] 更新状态块 `PHASE_0_PREFLIGHT: DONE`，进 Phase 1。

---

## 3. Phase 1 — 8×8 afterstate-V 网络 + FVI 训练基础设施（新文件 `rl8.py`）

> **与 4×4 的关键差异**：8×8 reachable 不可枚举（无法像 4×4 那样全 sweep 41503 个 board）→ 训练状态必须**采样**（self-play / rollout buffer），FVI 在采样到的 afterstate 上做半梯度回归。这是本阶段最大的新工程，须显式设计。

### 3.0 迁移 gate（**致命前置，复审 #1**）—— 先做，过了才碰 8×8
> 4×4 双 gate 是 `rl4.py`（**MLP + 全枚举 FVI**）认证的。`rl8.py` 把架构(CNN)、状态采样(buffer)、损失(变换)**三者全换** —— 这套新代码从未在可验真的地方证明过自己。
- 用 `rl8.py` 的 **8×8 CNN + 采样 buffer + 选定变换**，在 **4×4 引擎**（dp4 的单块/回合）上跑无折扣 mode-T，对照 `backward_dp_T(8)=3.5934 / (16)=4.8637`，**rel < 5% 才放行 8×8**。
- **4×4 喂 8×8-CNN 的适配（写死，否则 gate 测的不是同一 pipeline）**：conv 用 **padding='same'（全卷积，输出空间维 = 输入维）**；4×4 棋盘 **zero-pad 到 8×8 左上角**（pad 区 = 空格，是合法空盘格，无伪影特判），手牌/combo/k 编码维度与 8×8 完全一致 → flatten 维度一致。迁移 gate 与 8×8 训练**必须走同一份网络定义 / forward / 编码代码路径（同一 `rl8` 类，仅在 4×4 引擎上重新训练权重）**，**不得**为 4×4 另开 head、改输入维度或换 conv 配置——否则即便 gate 过也不代表 8×8 pipeline 正确。（注：gate 是**重训**一套 4×4 权重来验架构+采样+变换代码，不是迁移 8×8 权重——bdp_T(8/16) 是 4×4 mode-T 真值。）
- 这同时顺带验证了变换实现无低级 bug（但**不**验证重尾鲁棒性，那靠 §3.6 的 8×8 held-out 检）。
- **USER_GATE**：迁移 gate FAIL → 8×8 代码本身有问题，停下报我，**绝不**带着未认证的 pipeline 上 8×8。

### 3.1 状态编码
- 8×8 棋盘面（64 bit → 8×8 平面）+ 3 手牌 glyph 各一平面（手牌未放的标记）+ combo 当前值（分桶 one-hot 或标量，见 F1）+ rounds-left k（归一化标量）。
- afterstate = **3 块全放完后的棋盘**（combo 已贯穿），与 `beam_hand` 返回的 (board, combo, score) 同粒度（rl_plan 已验同粒度）。

### 3.2 网络
- 小 CNN：2–3 个 3×3 conv（32–64 ch）→ flatten/concat（combo+k 标量）→ MLP → 标量 V。~50–200k 参数。CPU 即可（MPS 对小 CNN 收益有限，先 CPU，若每 sweep 太慢再试 MPS）。

### 3.3 目标（mode-T，无折扣有限-horizon）
- `y(s,k) = mean_p max_pos( cl + V(s', k−1) )`，`V(·,0) ≡ 0`，**无折扣**；`mean_p` 对 3 块手牌组合（或采样手牌）平均；放不下 → 该步 0 continuation（与 channelB / bdp_T 同约定）。
- **目标里不含 heuristic**（V 已编码未来，加 heuristic = 重复计数）。

### 3.4 训练状态采样（8×8 专属，须落地；复审 #6 分布漂移）
- 维护 rollout buffer：用当前 V（贪心 `hand_score+V`）+ ε 探索跑 self-play，收集途经的 (board, k) afterstate。
- 覆盖 k 全谱（k=1..T，逐层 backward 填值像网络版后向归纳；low-k 与 high-k 都要有样本）。
- **on-policy 分布漂移防护**（复审重点）：V 只在自己贪心访问的态上训 → competence eval 时 strong/seer 走不同轨迹访问的态上 V 可能崩。**对策**：(a) buffer 混入 **strong-rollout 与 seer-rollout 轨迹态**做 off-policy 补采（不只 self-play）；(b) 预注册 `BUFFER_COVERAGE_τ`——eval 态（含 strong/seer 触及态）落在 buffer 支撑外的比例须 < τ 才放行 Phase 3。
- **报覆盖诊断**：buffer 的 board 密度 / combo 分布 vs seer 触及域的越界比例 + eval 态越界比例。

> **USER_GATE**：eval 态越界比例 > `BUFFER_COVERAGE_τ`（V 在被评估的态上没训过）→ 停，补采或报我。

### 3.5 warm-start
- `heuristic8(board)` 仿射映射到 V 量纲当 init 预拟合 1–2 epoch（a>0）；heuristic **只进 init 不进 target**。
- fallback：epoch-1 MSE 爆炸（> 阈，4×4 教训：大网用绝对阈而非 10× 相对阈）→ 冷启（FVI 对 init 不敏感）。

### 3.6 预注册：F1 combo 重尾处理（**Phase 1 定死，复审 #2 已推翻原方案**）
- combo 奖励无界 → 无折扣有限-T 的 V 在存活密集态可达 1e4–1e5 重尾，回归病态。
- **默认 = (c) 预测 per-round-rate × k**（复审推翻原"Huber-on-raw"：Huber 只抗离群点梯度，**不压 1e4-1e5 值域**，大目标几乎等权、早期低分态相对误差被淹没 → 仍病态）。per-round-rate 让每轮分稳定在 O(10²)、×k 还原，真正归一化值域。Huber 可叠加作辅助 robust loss，但不作主手段。
- **rate vs 总值 backup 写死（复审追问，混用必错）**：网络头输出 per-round-rate `rate_net(s,k)`，但 §3.3 的 Bellman 递归在**总值**域做。实现必须严格三步：(1) 取 continuation 先**还原总值** `V_total(s',k−1) = rate_net(s',k−1) × (k−1)`（`k−1=0` 时 `V_total ≡ 0`）；(2) `y_total(s,k) = mean_p max_pos( cl + V_total(s',k−1) )`；(3) **回归目标 `y_rate(s,k) = y_total(s,k) / k`**，对 `rate_net(s,k)` 做 MSE/Huber。**绝不**把 `rate_net` 直接当总值塞进 `max`（值域差 k 倍，必错）。competence/EVPI 下游用 V 时一律先 ×k 还原成总值再比。
- **微检（移到 8×8，不在 4×4）**：4×4 reward 有界、**触发不了重尾**，4×4 identity 检只能抓低级 bug、证伪不了"重尾下是否仍病态"。改为：在 8×8 上取一小批 **seer 触及的高分态**做 held-out，对比三变换的回归误差/收敛性 → 选实测最稳的。（4×4 的实现正确性已由 §3.0 迁移 gate 覆盖。）
- 报 combo cap 越界比例、各变换 held-out 误差。

#### 3.6★★ 用户治理覆盖（2026-06-02，strength-first，优先级高于下方 3.6★）
> 用户决定：**最强策略是真正交付物**，下方的三变换自洽选秀过重。落定：
> - **logv 已定盘**（机理无 ÷k/×k + 補13 隔离测尾部 rel 24-28% ≫ rate 57-66%），**不再办 rate/lograte 三方选秀**；rate/lograte 由 ×k 机理 + 補10/補13 证据直接淘汰。
> - **"最强"的真裁判 = Phase 3 实战比分**（rl-greedy-on-logv-V vs strong，同种子 CRN 配对，看谁得分高 + TOST），**不是** V 的校准。
> - **高-k 自洽（3.6★(c)）降级为 sanity sidecar**：训练后跑**一次**，只为给目标线② 的"天花板可信度"背书（V 没瞎报），**不当 blocking gate**；其 rel 阈值仅作 advisory（高了写进局限，不阻塞定盘）。
> - 逃生分支 (d) 仍在：若 logv 训练发散/无法收敛 → USER_GATE（降 T / cap combo）。
> 下方 3.6★ 的 (a)(b)(d) 仍成立；只有 (c) 从"blocking 真裁判"降为"sanity"。保留原文作预注册记录，不删改。

#### 3.6★ 变换定盘规则（2026-06-02 用户审计后**重写并预注册**，写后不许事后改 — §0.9）
> 旧 NEXT_ACTION 的"训到 T=3、比锚、选 rel 最小者"**作废**。理由（用户上一轮审出的真洞）：`rate` 与 `lograte` 的 decode **都含 ×k 放大器**（`rl8.py:288-294 decode_vtot`），高 horizon 把网络小误差放大成大偏置（lograte 在 4×4 gate T=16 即 FAIL 11%，补10）；而 `exact_vtot_anchor` 只能精算到 **T=3**，生产 horizon 是 **T=50**，×k 放大量在 T=50 比 T=3 约大 **17 倍** → 一个在 T=3 锚上 rel 最小的变换，到 k=50 照样可能爆。"比 T=3 锚"根本**区分不出** logv（无 ×k、机理免疫）与 rate/lograte（被放大）。故定盘规则改为以下四条：

- **(a) 机理优先 `logv`**：`logv = log1p(V_total)`，decode `expm1(u)`，**无 ÷k/×k 放大器**（`rl8.py:292-293`）→ 唯一不随 horizon 线性放大网络误差的候选。logv 是默认前锋（补13：满覆盖 T=3 隔离测 logv 尾部 rel 24-28% ≫ rate 57-66%）。
- **(b) T=3 精确锚降级为"低级-bug 证伪器"**：`exact_vtot_anchor`(T=3) **只**用来抓 encode/decode 实现错（某变换连 T=3 都大偏 = 有 bug），**不再当唯一/最终裁判**——T=3 horizon 太短，测不到 ×k 在高 k 的放大。
- **(c) 真裁判 = 高-k 自洽检验（V_net vs MC-rollout，接近 T=50）**：MC-rollout 通过**真实发牌、贪心-on-V_net 实玩 k 轮、累加真实得分**估 `V̂^π(s,k)`——**全程不经 decode ×k → 对 ×k 放大免疫的无偏参照**（标准 rollout policy evaluation / Bellman 自洽，见 Tesauro rollout、arXiv:2504.02221）。预注册判据：
  - **探针集** = empty + **32 个冻结**中局 (board,combo) 态（从 **strong+seer 混合 rollout** 采，off-policy，§3.4，避免只测 self-play 可达态），在 **k∈{25, 50}** 求值。
  - 每探针 MC：**M_sc = 2000** 条未来发牌序列；报 16-block-SE / paired-bootstrap CI（§0.3，禁 ratio-of-means）。
  - 自洽相对误差 `rel(s,k) = |V_net(s,k) − V̂^π(s,k)| / V̂^π(s,k)`（V_net 已 decode 成**总值**）。
  - **PASS（两条都满足）**：① 探针集聚合中位 `rel @ k=50 < 0.10`（8×8 重尾 + MC 噪声，比 4×4 gate 的 5% 放宽一档）；② **无-×k 放大签名**：`rel@k=50 / rel@k=25 < 2.0`（logv 这类免疫变换 rel 随 k 近持平；×k 变换 rel 随 k 近线性增长，k 翻倍 rel 约翻倍 ≥2 → 判定为有放大器，FAIL）。
- **(d) 预注册逃生分支（防默默掉进 inconclusive）**：若 **logv / rate / lograte 全部**在 (b) 锚 + (c) 高-k 自洽上**失败** → **停下报用户 USER_GATE 重设计**（候选：降 T / cap combo / 改 mode-γ 近似），**绝不**默默记为 inconclusive 继续跑。

> **USER_GATE（§3.6 定盘）**：(d) 触发（全候选失败）；或选定变换在高-k 自洽上 `rel@k=50 ≥ 0.10` 或放大签名 `≥ 2.0` → 停下报用户。

### 3.7 预注册：位移检查（防"V 没动假装收敛"）
- 训练后探针集 V 变化（V-单位）> `DISPLACEMENT_τ`。⚠️**去循环化（复审）**：8×8 无 V* 真值，**绝不**用"warm_init 与训练末 V 的 RMS 差"做阈——阈值取自被测的同一 final-V，V 动得多阈也跟着大，几乎恒 PASS、丧失"V 没动"的检测力。改用**不含 final-V 的量纲**：`τ_disp = 0.3 × RMS(warm_init 探针值自身)`（纯 init 统计，与 final 无关），或锚定到 competence 缺口量纲（strong−greedy 的每态 V 差中位数）；二者取其一，Phase 1 **仅用 warm_init / heuristic 统计**预注册填死，填后不许改。
- 辅判据 corr(V_net, heuristic8) 不作硬 FAIL，但过高（如 >0.95）触发人工核查"V 是否只是仿射缩放启发式"。

**完成判据**：`rl8.py` 能在 4×4 引擎跑通 F1 微检（PASS）+ 8×8 单 sweep 跑通且 print 进度。实测单 sweep wall-clock → 回填状态块 `WALL_CLOCK_CAP_TRAIN`，并**立即按 HARD_CEILING 规则外推**（单 sweep × min_sweeps × 1.5）：外推 ≤ 7 天才允许标 `PHASE_1_INFRA: DONE` 并进 Phase 2；外推 > 7 天 → 标 `PHASE_1_INFRA: BLOCKED_BUDGET`，**不进 Phase 2**，走下方 USER_GATE。

> **USER_GATE**：F1 微检在 4×4 上改变了双 gate 结果（变换实现有 bug）；或**外推总训练 > HARD_CEILING(7 天)**（必须在开训前停，不许带着超预算的 plan 进 Phase 2 跑到撞墙）；或状态采样覆盖严重不足（combo 域越界比例过高）→ 停下报我。

---

## 4. Phase 2 — 训练到 plateau（后台，预算受控）

- 跑 mode-T FVI 训练到 `PLATEAU_RULE` 命中或撞 `WALL_CLOCK_CAP_TRAIN`。
- `min_sweeps ≥ ~T_MAX` 再允许 plateau 判据生效（4×4 教训：mode-T 是网络版后向归纳，早停 = 严重 undershoot）。
- 每 sweep 打印探针集 V、MSE、plateau 指标、Huber 削顶比例（全 `flush=True`）。
- 产出训练好的 `V_net` checkpoint + 训练曲线 JSON。

> **USER_GATE**：撞 wall-clock 上限仍未 plateau（援引"inconclusive 逃生门"前必须报我，不许自行判定）；MSE 发散/NaN；位移检查 FAIL（V 没离开 init）→ 停。

---

## 5. Phase 3 — competence gate（vs beam-strong，CRN 配对）

- 新 `play_rl`：同 `deal-{seed}` 流，每手 `beam_hand` 候选 argmax `hand_score + V(afterstate, k)`，**无前瞻**，替换 `strong` 的位置。
- 在 `EVAL_SEEDS`（冻结、与训练不相交）上跑 rl vs strong **paired-CRN**，固定 horizon T=50（主报）+ {40,60}。
- 按 `COMPETENCE_TEST`（**TOST 等价检验** + 预注册 margin，**非"CI 含 0"**）判 ≫ / ≈ / ≪。报 `rl − strong` 均值差 + paired CI + 16-block-SE。
- **正交性诊断（复审 #4）**：rl/strong/`beam_hand` **共用同一 beam 候选枚举 + 同一 heuristic**（fast.py:108/127/132/136），只在 value 估计层正交、**搜索/候选层同源**。须在**部分手上枚举 full action**（非 beam top-B）确认 beam 未系统性裁掉 RL 偏好的落点；若裁掉 → 打平可能是同源候选池假象。
- 同时报 rl vs seer/blind（为 Phase 4 EVPI 重算备数据，同 cohort 约定）。

> **USER_GATE**：CRN 配对被破坏（rl 与 strong 不同种子/不同发牌流）；TOST 落在"既不等价也不优"的 inconclusive 区；full-action 诊断显示 beam 系统性裁掉 RL 偏好落点 → 报我定性。

---

## 6. Phase 4 — verdict 路由（预注册结果表，三选一，不许事后挑）

- **(i) RL ≫ strong**（CI 下界 > 阈）→ 天花板被低估，EVPI 占比是高估。**在同一冻结 intersection-of-survivors cohort 上**用 RL 当新天花板/分母**重算 EVPI 占比**（硬约定 6：不用 RL 自己的存活集）。预期占比**只会升**。报新区间 + CI。**headline：目标线①拿到更强策略 + 目标线②占比上修。**
- **(ii) RL ≈ strong**（TOST 判等价：90% CI ⊂ [−Δ_eq,+Δ_eq]，且 §3.0 迁移 gate 过 + greedy 被碾压 + V 已离开 init）→ 天花板被两方法佐证，**headline 谨慎加强**。⚠️**措辞降级（复审 #4）**：rl/strong 共享 beam 候选枚举 + heuristic，正交性**仅在 value 估计层**（learned-V vs heuristic-greedy），**搜索/候选层同源**。**不得宣称"独立验证"**；只能写"在同一候选搜索框架下，学习型 value 与手工启发式 value 给出一致天花板估计（共享候选枚举是已声明的局限）"。
- **(iii) RL ≪ strong 但过 4×4 gate** → "预算内未达搜索天花板，inconclusive"，**永不当结论**；记录为"已知局限：8×8 学习型逼近器在本预算下未超越 beam-strong"。

> **USER_GATE**：进入 (i) 重算 EVPI 时，务必核对 cohort 是 intersection-of-survivors 且 D=3 N=120 显式传参 → 重算前报我确认 cohort 定义。任何想把 (iii) 写成"RL 证明 strong 是最优"的措辞 → 停（那是过度声明）。

---

## 7. Phase 5 — figures + 记忆/文档回写（无 commit）

- 出图：rl vs strong 收敛配对图 + （若 (i)）EVPI 占比上修对比图，零依赖 SVG 风格同 `plots_oracle.py`。
- 回写：`memory/` 新建 `rl8_results.md`（原子化 ≤150 行：winner 配置 + competence 结果 + verdict 分支 + 工程教训）+ 更新 `MEMORY.md` 索引 + `log.md` 追加 + `daily/`。更新 README 的"方向②a"节与 GOAL/ROADMAP 的对应 checkbox。
- **不 commit / 不 push**。

> **USER_GATE（强制）**：Phase 5 完成后，把"要 commit 的文件清单 + diff 摘要 + 拟用 commit message"摆给我，**等我明确说 push 才推**。

---

## 8. 已知不认证 / 诚实边界（写进 README，别吹）
- 8×8 RL competence gate 认证的是"学习型逼近器 vs beam-strong 的相对强弱 + 由此对 EVPI 占比的影响"，**不**等于"找到了 8×8 真·最优"（8×8 不可精确解，真最优永不可知）。
- (iii) inconclusive 永远不能反向解读为"strong 已是最优"。
- 57–69% 即使在 (ii) 也仍是相对"当前最强已知在线玩家"的下界（更强玩家会再抬占比）。

---

## 9. 审查溯源
- 本计划综合：`memory/rl_plan.md`（RL 四轮审核定稿）、`PHASE_EXEC_PLAN.md §②(a)`、`NEXT_PHASE_PLAN.md §3(a)`、`memory/oracle_immortality_reframe.md`（六大陷阱）、`rl4_gate_results.md`（4×4 教训：FVI+max 高估偏置随网宽降 / mode-T 早停陷阱 / 大网 warm fallback 过敏）。
- 2026-06-01 系统审查 A–F 维度结论已并入（A 三引用、B 的 dp4 两小修、F 拍板 ②a 先做）。
- 本计划经 1 轮独立 adversarial 复审（见 Phase 0 前的 review 记录）。执行中仍遵守"实质方案先独立审核"的项目约定。
