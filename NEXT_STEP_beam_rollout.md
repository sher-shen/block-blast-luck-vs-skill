# 下一步执行计划：beam-rollout 破 2950 天花板（2026-06-05 立，审核已过）

> 在**新对话**里独立执行本文件。目标线①：造 8×8 Block Blast 最强不预知未来策略。
> 本计划 = 验证"把 lookahead 的 rollout 基策略从弱 1-ply greedy 换成强 beam 基"能否突破
> ~2950 的结构性天花板。**这是诊断认定的唯一可能破墙杠杆。**
> 背景/诊断见 memory/strongest_policy.md；时间线 memory/log.md 续14+续15。

---

## 0. 开工前必读（已由独立审核坐实的事实，别重复验证）
- **杠杆活着**：log 续14 那条 "base=cem rollout 反降 2663" 用的是 **1-ply cem-greedy 基**
  （`cem.py:177-195` `_rollout_leaf` 对 `base in {heur,cem}` 都是 `for pid in future[i:i+3]` 逐块 1-ply），
  **不是 beam**。所以"换强 beam 基"这个杠杆从未被真正测过 → 值得测。
- **2663 带双重 confound**：弱 1-ply 基 + CEM 权重 `w`(models/cem_w.json, `B_train=8` beam 训练)被当 1-ply 用 = off-policy。
- **算力便宜**：M=200 D2S30 base=heur ≈ **100s**（32 核 ProcessPoolExecutor）。全程分钟级，不是小时级。
- **真正风险 = off-policy 叶-V 失配**，不是算力 → 隔离变体 A/B 是本实验的灵魂。

## 环境 & 纪律（硬约束，违反作废）
- `.venv\Scripts\python`；`PYTHONIOENCODING=utf-8`。后台任务 cwd 重置回 c:\，须显式 `cd C:\Users\sher\block-blast-luck-vs-skill`。
- **torch 只在 rl8/rl4；cem.py 保持 torch-free**（import fast，位运算特征 + linval）。
- 评测：**CRN 配对 + 16-block-SE**（`cem.py:349-355` `_block_stats`、`358-366` `_paired_ci`，2000 boot，只用 diff）；
  **禁 ratio-of-means**；**判强一律用实战 rollout 分**（不是训练 elite_mean）。
- 训练流 `cem-train-{gen}` / 报告流 `bench` tag 分离防泄漏（`gen_streams`，`cem.py:234`）。
- **所有 commit/push 先问用户。** 现工作树未提交：cem.py / models/cem_w.json / cem_*.json。

---

## 1. 需要的代码改动（cem.py，低难度）
1. **`_rollout_leaf` 加 `base=="beam"` 分支 + `Br` 参数**：当 `base=="beam"` 时，每"手"(3 块)不再逐块 1-ply，
   而是用 `cem_hand` 同款联合 beam（宽 `Br`，连击贯穿）铺整手，取最优末态续滚。
   - 复用 `cem_hand`（`cem.py:115-136`）的结构；把它抽成一个"按手 beam、返回 (加分, board, combo)"的小函数，
     `_rollout_leaf` 在 `base=="beam"` 时按手调用、累加分。`base in {heur,cem}` 路径保持原样不动（防回归）。
   - 也可加 `base=="hbeam"`（用 `fast.heuristic` 当 beam 排序键 = strong-beam 基，规避 off-policy `w`）。
2. **叶尾值开关（隔离变体）**：`play_cem_look`（`cem.py:198-215`）里叶子 `linval(末板, w)` 加一个 `use_leaf` 开关。
   - 变体 A：`use_leaf=True`（现状，留叶 V）。
   - 变体 B：`use_leaf=False`（砍叶 V，仅返回 rollout 累计分）；配合可加深 D 吃尾巴。
3. **CLI 参数化**：`bench_look` / `perseed` / `pair` 现在 D/S/B/base/cand 多为硬写。
   给入口加参数解析，支持 `python cem.py look M T D S B base cand [Br] [use_leaf]`。难度低（审核给过示例）。
4. 改完先跑 `python cem.py smoke` + 一个 M=8 的 look，确认无回归、落点还原 0 mismatch。

---

## 2. 实验序列（fail-fast，每步看信号决定是否继续）

### Step 0 —— 近免费的天花板交叉检查（先做，~10 min）
- **B-sweep**：`look M=50 D2 S30 B∈{12,24,48} base=heur cand=cem`。若 B 也饱和（像 S30 那样）→ 排除"候选末态宽度不够"，
  强化"墙在续法/视野"。若 B 没饱和 → 那才是更便宜的破墙点，**先去拓宽 B，本计划暂缓**。
- **补持久化基线**：把 2663（D2 S30 base=cem，1-ply）和 D3（log 说反降但没存数）真跑一遍存 json，作对照锚点。
- **检查候选退化**：B=48 时看 bench_look 是否出 "not cand"/合法末态过少的警告。

### Step 1 —— 核心：预算中性 S→base 换算（M=50 先出信号）
- **对照**：D2 **S30** B12 base=**heur**（= 现交付 2917 的同流复现）。
- **处理**：D2 **S6** B12 base=**beam** Br∈{2,3,4}（砍 S 5× 抵 beam 成本 ~Br×，算力≈持平）。
- **两种 beam 基都测**：`beam`(linval/`w` 排序，有 off-policy 风险) 和 `hbeam`(heuristic 排序，无 off-policy)。
- **隔离变体 A vs B 同时跑**（决定性！）：
  - A：base=beam/hbeam，**留叶 V**（use_leaf=True）。
  - B：base=beam/hbeam，**砍叶 V**（use_leaf=False），可配 D3 补尾。
- **配对裁决**：`cem.py pair` 同流 per-seed 配对 vs base=heur 对照，给 d ± 16-block-SE + 95%CI（boot，禁 ratio）。

### Step 2 —— 只在 Step1 有正信号才加码（M=200 全量）
- 把胜出配置（某个 beam 基 + A 或 B）拉到 **M=200 CRN 配对** vs 现交付 2924/2918，出最终裁决。
- 若破 2950 显著（CI 不跨且 d>0）→ 更新交付 + 全景表 + 落点还原验证。

---

## 3. 判读规则（让负结果也决定性）
| 现象 | 结论 |
|---|---|
| 变体 A、B 都 ≤ base=heur（打平/降） | **真·结构天花板**，第二次独立坐实 → 收口，写诚实负结果（可发表，同 bsmc 叙事） |
| 仅变体 B（砍叶 V）回升、A 还降 | **2663 是 off-policy 叶失配假象**，杠杆真有效 → 走 Step2，并考虑重训 `w` 配 beam 续法 |
| 变体 A 直接涨、破 2950 显著 | 杠杆成立 → Step2 全量确认 + 升级交付 |
| `hbeam`(无 off-policy) 也不动 | 偏置不是瓶颈，强基策略救不了 → 收口 |

## 4. 收口 & commit（做完实验后，先问用户）
- 无论正负，把结果写入 memory/log.md 续16 + strongest_policy.md 全景表/下一步。
- **commit/push 必须先问用户。** 待提交：cem.py（含本次改动）/ models/cem_w.json / cem_*.json / 新 json。
  建议把已完成的 CEM 独立强基线先 commit（不 push），再 commit 本次实验产物。

## 5. 关键代码坐标（审核给定，省得重找）
- `_rollout_leaf`（rollout 1-ply，要加 beam 分支）：`cem.py:177-195`
- `cem_hand`（联合 beam，复用源）：`cem.py:115-136`
- `cem_greedy_hand`（1-ply 训练用，对照）：`cem.py:95-112`
- `play_cem_look`（叶尾值 linval，加 use_leaf 开关）：`cem.py:198-215`
- `bench_look` grid（要 CLI 参数化）：`cem.py:230-237`
- 16-block-SE / 配对 CI / 等价性：`cem.py:349-366,394-399`
- CEM 权重训练配置（B_train=8 beam）：`models/cem_w.json`
