# 换机接续提示词（新电脑 / 新对话从这里开始）

> 这个仓库是 block-blast 8×8「运气 vs 技能」量化项目。在新电脑上 `git clone` 本仓库后，
> 开一个新的 Claude Code 对话，把下面「提示词」整段粘进去即可无缝接续。
> 操作性 source-of-truth = `PLAN_8x8_RL.md`（状态块 NEXT_ACTION + §3.6★★ + §0 硬约定）。

## 0. 新电脑环境准备（粘提示词前先做）
```bash
# 1) clone（仓库名是 block-blast-luck-vs-skill；可指定成 block-blast-sim 省得改路径）
git clone https://github.com/sher-shen/block-blast-luck-vs-skill.git block-blast-sim
cd block-blast-sim
# 2) 装 venv：必须 python3.13（py3.14 无 arm64 torch wheel），torch 验证过 2.12.0
python3.13 -m venv .venv && .venv/bin/pip install torch
# 3) 冒烟验证：应打印 RateNet params=140897 + 候选数，无报错
.venv/bin/python rl8.py smoke
```
- 本机是 Mac M-series → rl8.py 顶部 DEVICE 自动用 MPS（快）；否则自动回退 CPU（能跑，慢 ~10×）。
- 训练 ckpt 在 `/tmp/*.pt`、**不随 repo**；新电脑从零冷启训练（可复现，小时级）。

---

## 1. 提示词（整段粘进新对话）

```
我继续 block-blast-sim 项目（已从私有 GitHub 仓库 sher-shen/block-blast-luck-vs-skill clone 到本机）。
先读这几个文件再动手，别预读别的：
- PLAN_8x8_RL.md（/loop source-of-truth：看「状态块」NEXT_ACTION + §3.6★★ 用户治理覆盖 + §0 十条硬约定）
- GOAL.md（权威目标）
- memory/MEMORY.md → 按需读 memory/rl8_phase1.md + memory/log.md 续12 补14~21（最近进展）

【真实目标，先对齐】
终极目的只有一个：把这游戏的 luck/skill 占比（现 57–69%）钉成可信的数。造一个不靠预知未来的最强 8×8 可玩策略是「必经的工具」——用它当新天花板撞「strong=在线天花板」这个假设、重算 EVPI。已转 STRENGTH-FIRST：判最强用「实战得分」（rl vs strong 同种子配对），不看 V 校准；高-k 自洽降级为 sanity。别引入「人类口诀/自动玩家发布」这类噪音。

【当前进度 / 卡点演变（重要）】
Phase 1 IN_PROGRESS。§3.0 迁移 gate 已 PASS。§3.6 变换历经：logv→rate 都在无折扣 T=50 训练发散（补17-20，根因=无折扣长程 mode-T 值迭代本质不稳，deadly triad 跨 50 层复利，与 4×4 当年同构）。**用户已拍板 γ-折扣训练**：rl8.py 已实现 GAMMA（默认 1.0 保 4×4 gate；8×8 train8/check8 自动设 0.95），γ 只当训练稳定器，判强/天花板仍用无折扣实战得分（rollout），γ 不进 channelB → §0.2 红线不破。γ=0.95 probe 已证增长被压、有界趋势（补21）；上一台机器的全程 γ 训练跑到 ~sweep70（v8=259，有界爬升、未 plateau）被中断换机。

【新电脑第一步：环境】
1) 装 venv：python3.13 + `pip install torch`（验证过 torch 2.12.0）。py3.14 无 arm64 wheel，必须 3.13。
2) `.venv/bin/python rl8.py smoke` 应打印 RateNet params=140897 + 候选数，无报错。
3) Mac M-series 自动用 MPS；否则回退 CPU。DEVICE 在 rl8.py 顶部自动判。

【接着执行（loop 从 NEXT_ACTION 接）】
① `.venv/bin/python rl8.py train8 rate 50`（用 γ=0.95 冷启全程训练，3h 一段、断点续 /tmp/rl8_8x8_rate_T50.pt；没 plateau 就重复这条续训）。后台运行用 run_in_background 与 nohup& 二选一，绝不叠加（§0.5）。
② **决定性观察**：v8(=Vtot(0,8)) 是否 plateau 在「有界 sane 值（~800-1000）」而不是顶到 clamp(u≈5000，表现为 v8≈3万+、v8≈v16/2 但量级爆）。
   - 有界收敛 → 建 competence_gate（新函数，复用 rl8.py 的 mc_rollout_value 骨架：rl-greedy-on-V vs strong，冻结 EVAL_SEEDS、同种子 CRN 配对、固定 T=50、报 paired 得分差 + TOST + 16-block-SE）= 最强策略 verdict。
   - 还是顶 clamp（理论上不该，γ-收缩有保证）→ USER_GATE 报我：降 γ(0.9) / 降 T / 换纯 value 头。
③ 顺手 `.venv/bin/python rl8.py check8 rate 50 2000 32` 跑一次高-k 自洽当 sanity（量化 ×k 偏置=目标②天花板局限，advisory 不阻塞）。
④ HARD_CEILING 终判 → PHASE_1_INFRA=DONE → 按 verdict 路由 Phase 4（rl≫strong=占比上修 / rl≈strong=佐证 / rl≪strong 但过 gate=inconclusive 局限）。

【硬约定】严守 PLAN §0 十条（torch 只在 rl8/rl4；无折扣有限-T 用于对齐 channelB——γ 只做训练稳定器不进比较；带 CI 禁 ratio-of-means；诊断前 rm -rf __pycache__；print flush=True；后台 run_in_background 与 nohup& 二选一绝不叠加；cohort 用冻结 intersection-of-survivors；eval 种子与训练不相交）。
所有 commit/push 必须先问我（USER_GATE）。每命中 USER_GATE 停下问我。
```

---

## 2. 关键文件地图（新对话按需查，不预读）
- `PLAN_8x8_RL.md` — /loop 执行计划（状态块 + NEXT_ACTION + §3.6★★ + §0 硬约定）
- `GOAL.md` / `ROADMAP.md` — 权威双目标 + 阶段路线
- `rl8.py` — 8×8 afterstate-FVI（CNN + 采样 buffer + GAMMA + mc_rollout_value + selfcons_highk + offpolicy_states；CLI: smoke/gate/train8/check8）
- `rl4.py` / `dp4.py` — 4×4 双 gate + 精确 DP 锚
- `fast.py` — bitboard 引擎 + strong/beam/lookahead
- `memory/MEMORY.md` — 记忆索引；`memory/log.md` — 时间线（最新补21）；`memory/rl8_phase1.md` — 工程教训（×k 放大 / 高-k 自洽 / OOM / γ）
