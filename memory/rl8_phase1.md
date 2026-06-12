# rl8 8×8 afterstate-FVI — Phase 1 状态 + 工程教训（2026-06-02，进行中）

> 方向②a 的 8×8 RL（`rl8.py`，仅此+rl4 依赖 torch）。本文件 = 当前状态 + **可复用工程教训**。
> 完整时间线见 [log.md](log.md) 续12 补1~补11。最终 winner/verdict 待 Phase 5 写 rl8_results.md。
> 操作性 source-of-truth = `../PLAN_8x8_RL.md` 状态块 + NEXT_ACTION。

## 当前状态
> ⚠️ **终局更新（2026-06-03/04，取代下面"截至 2026-06-02"快照）**：logv（expm1 爆）与 rate（线性爬顶）
> **都在无折扣 T=50 训练发散**（log.md 补17–20）→ 确诊**非变换问题，是无折扣长-horizon mode-T FVI 本质
> 不稳**（deadly triad 跨 50 层复利，与 4×4 dp4 当年同构）。用户拍板 **γ=0.95 只当训练稳定器**（判强/
> 天花板仍用无折扣实战得分，γ 不进 channelB → §0.2 红线不破）→ 干净 plateau（补21/续13）。
> competence gate：纯 greedy-on-V **RL≪strong**（1832/1888 vs 2378，rl8_competence*.json；根因=OOD 高估
> 早死，B=200 放开候选池暴跌坐实稳健非池假象）→ **on-policy 策略迭代修好 = 2353 ≈ strong**（TOST 等价，
> rl8_competence_pi.json）→ **价值引导前瞻 vla D2 S30 = 2924±49 = 最终交付**（+25% vs strong）。
> 后续与"2950 重框"见 [strongest_policy.md](strongest_policy.md) + log.md 续13–18。

### （历史快照，截至 2026-06-02）
- **Phase 0 DONE**：dp4 默认 γ 0.99→0.95(复现 3.9157)、移除单窗 assert 字段、README 补 3 引用。
- **§3.0 迁移 gate PASS**：rl8 的 8×8-CNN + 采样 buffer + FVI 在 4×4 复现 bdp_T(8)=3.5934/(16)=4.8637，**T8 1.04% / T16 2.74%**（rl8_gate.json）→ pipeline 架构认证（用 transform='rate'）。
- ~~**§3.6 变换未定盘**（Phase 1 唯一卡点）~~（已终局，见上）：见下「×k 放大陷阱」+「高-k 自洽真裁判」。combo 接线已完成。当时前锋=logv（补13）；定盘规则按用户审计重写（补14）。
- **8×8 预算可行**：单 sweep ~15-19s(MPS, T=50, hidden=128)，收敛外推 ~13-18h ≪ HARD_CEILING 7 天。

## 工程教训（可复用，非显然——未来 8×8 RL / CNN-FVI 直接套）

### 1. ⚠️ ×k decode 放大陷阱（本项目最深，独立复审挖出）
- per-round-rate 把网络输出 u 解码成 `V_total = u×k`。**任何 net 误差被 ×k 放大**：高 horizon k=16 时 0.02 的 u 误差 → 11% 的 V 误差。
- `lograte`(decode `expm1(u)×k`) **同病**——micro-check 因其 log 空间 loss 小（假象，只因目标小）误选它，但它 4×4 gate **T=16 偏低 11% FAIL**。
- **`rate`(V=u×k) 也有 ×k**，只因 4×4 reward 有界、per-round rate 小才不暴露 → "rate 过 4×4 gate" **不**等于 rate 在 8×8 重尾下无偏。
- **教训**：有界的验证引擎（4×4）测不到重尾域；选变换**必须**在重尾域验正确性 → 见教训 5 的精确锚。候选解 `logv=log1p(V_total)`（decode `expm1(u)`，**无 ÷k/×k**）去掉放大器+压尾，待验。

### 2. 训练集必须取前向闭包，不能只用 self-play buffer
- FVI 半梯度：续值在 dst 态上 bootstrap。若只训 self-play 采到的 src（小），dst 集（大得多）上的续值靠外推 → **值崩塌成"只剩即时分"**（4×4 实测 src=930/dst=20072 → V 从 1.08 掉到 0.79）。
- 修复：训练集 = buffer 的**前向闭包**（4×4=全 reachable 41503，续值全覆盖）。8×8 不可枚举 → 训采样 buffer，覆盖率由 BUFFER_COVERAGE_τ 单独 gate（Phase 3）。

### 3. MPS 10× 加速 + 两个坑
- CNN 全 reachable full-batch FVI：CPU=2187ms/fwd+bwd（→180s/sweep），**MPS=211ms（10×）**→14s/sweep。小 CNN 也值得上 MPS。
- 坑①：board plane 逐格赋值**必须在 CPU 构建再整批 `.to(DEVICE)`**——MPS 上逐格赋值是上千次慢分发。
- 坑②：probe 的 `torch.zeros/full` 必须显式 `device=DEVICE`（否则 input CPU/权重 MPS 报错）。`scatter_reduce(amax)` MPS **支持**。
- 坑③（补16）：FVI 的 **full-batch eval forward** 不随 buffer 长大缩放——buffer recollect 后 eval-dst→~1M，单批 conv 激活 (N,32,8,8) 要 11.5GiB → MPS OOM。修=`net_forward_chunked`(分块 120k) + 8×8 训练关 recollect(固定 buffer)。

### 4. 冻结每-sweep 目标（防 k 轴 interference）
- 一个 sweep 内逐 k 边算目标边训 → 共享权重的网络在 k 轴上 catastrophic interference（训 k=16 扰动 k=1）。
- 修复（rl4 式）：先用**冻结 net** 建全 k 目标，再训 inner_steps 遍。配断点续训（save/load state_dict）让长收敛跨 session。

### 5. exact_vtot_anchor — 不可枚举引擎的真值验证法
- 8×8 无全局真值，但训练用**固定 8 hands** → 其 mode-T 后向归纳可**精确算**（非 MC，memo (board,combo,k)）。
- `rl8.exact_vtot_anchor(engine, hands, T)`：V_total(empty,k)={1:11.0, 2:65.9, 3:110.6}（T=3 精算 52s）。k1→k2 暴涨（11→66）证锚确实压重尾域。
- **这是第一个能在 8×8 重尾域验变换正确性的靶**——选变换靠它的 rel 误差，不靠"哪个 loss 小"。

### 6. 杂项
- mode-T backward induction：**T=8 比 T=16 先收敛**（层数少），T=16 滞后需 resume 续训（4×4 rate gate 共 ~550 sweep 才 T16<5%）。早停 = undershoot。
- FVI 收敛值**震荡** ±0.3 → 判 PASS 看 trailing-mean 不看单 snapshot。
- 全程**本地 Mac MPS**，不上集群（集群是 dockingmc 线）。

### 7. ★高-k 自洽真裁判：MC-rollout 是对 ×k 放大免疫的无偏靶（补14，用户审计驱动）
- 教训 1/5 的精确锚 `exact_vtot_anchor` **只精算到 T=3**，但生产 horizon T=50。×k 放大量∝horizon → T=50 比 T=3 大 ~17× → **T=3 锚根本区分不出 logv（无×k）vs rate/lograte（被×k）**。"训 T=3 比锚选最小"是会骗人的判据。
- **正解 = 高-k 自洽**：MC-rollout（贪心-on-V_net 真实玩 k 轮、累加真实 hand_score）估 V^π(s,k)。**返回值是真实累加分，全程不经 decode ×k**（decode 只在 argmax 选点用）→ 对 ×k 放大免疫的无偏参照（标准 rollout policy eval / Bellman 自洽）。在 k≈50 比 `rel=|V_net−MC|/MC`：① 绝对 rel<0.10；② **无-×k 放大签名 rel@k50/rel@k25<2.0**（免疫变换 rel 随 k 持平，×k 变换 rel 随 k 近线性增长）。
- **smoke 验证签名检测器工作**：4-sweep 欠训 logv，empty k50 V_net=119 vs MC=2338（rel0.90 欠训），但 **rel@k50/k25=1.06≈flat → 确认 logv 无×k 放大**。即使欠训也能读出"有无放大器"这一结构属性。
- **公平性前置**：训练 buffer 须含与探针同分布的 strong+seer off-policy 态（`train_modeT(offpolicy=(ns,nz))`）→ 否则自洽检验测的是【训练覆盖】而非【变换是否被放大】。
- 工程：`board_planes_fast8`（split-32bit bit-unpack 避 signed-int64 bit63 溢出）让百万级 rollout board 编码可行（逐格 python 太慢）。
