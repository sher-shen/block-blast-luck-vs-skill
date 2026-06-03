# 方向① 4×4 afterstate-FVI 双 gate 认证结果 + 工程教训（2026-06-01 续9）

> `rl4.py`（唯一依赖 torch 的文件）。结果 `rl4_gate.json`。**双 gate 皆 PASS ⇒ 认证 4×4 pipeline，放行 8×8（②a）。**

## winner 配置（现码现跑可复现）
- 环境：py3.13 venv + torch 2.12.0（**py3.14 无 arm64 wheel，必须降 3.13**）。CPU（MPS 对小 MLP 无收益）。
- 网络：MLP in→hidden→hidden→1，ReLU 无 BN。**mode-γ hidden=256；mode-T hidden=128**。
- FVI：向量化半梯度（scatter_reduce amax 做 per-(b,p) 段最优落点，段 init=0 自动实现 max(0,·) 与"放不下→0"）。reachable N=41503，transitions=1.67M，build 2s。
- warm-start：heuristic4 仿射映射到 V* 量纲 [0,4.6]（a>0）；只进 init 不进 bellman target。
- 训练：mode-γ 310 sweep/214s plateau；mode-T 102 sweep/637s plateau。inner_steps=20，lr 1e-3（@300/200 decay×0.3），plateau_tol 0.1% over 8 sweep，min_sweeps 60/80。

## gate 结果
| gate | 判据 | 真值 | 结果 |
|---|---|---|---|
| γ-值 | \|V_net(empty)−3.9157\|<0.05 | 3.9157 | PASS Δ=0.0013 |
| γ-策略 | greedy-on-V_net 最优比 paired-CRN bootstrap CI 下界≥0.92 | 近视贪心 0.859 | PASS ratio 1.005 CI[0.997,1.013] |
| T-值 | T∈{8,16} rel<5% | bdp_T(8)=3.5934,(16)=4.8637 | PASS 2.2%/3.9% |
| T-在线 | M=50k 无折扣 rollout rel<5% + 护栏 2·SE<2.5%·bdp_T | bdp_T(T) | PASS 1.6%/1.5% 护栏过 |
| 位移 | disp(V-unit)>τ_disp binding；corr<0.92 advisory | corr(heuristic4,V*)=0.870 | PASS disp0.74>0.22; corr0.871 |

## 工程教训（reusable，非显然）
1. **FVI + max + 函数逼近 = 系统性高估偏置，偏置随网宽单调递减**。hidden=64 → V(empty) +3.9%（FAIL 紧 gate）；256 → +0.03%（PASS）。机理：V[nb] 有逼近噪声 ε，max_pos 偏选 ε>0 的落点 → Jensen 间隙逐 bootstrap 累积到根节点。**压偏置的唯一干净手段=减逼近误差=加网宽**（不是加训练步——固定点偏置，训练再久不动）。
2. **网宽 = accuracy↔wall-clock 旋钮，按 gate 紧度选**。紧绝对值 gate（γ 的 0.05）→ 大网（256）。松相对 gate（T 的 5%）→ 小网（128）够用且每 sweep 快 ~2.5×。256 跑 mode-T 撞 1800s wall budget 仍欠收敛；128 干净 plateau。
3. **mode-T = 网络版后向归纳，值逐层 k=1→16 传播，早停是陷阱**。首版 sweep18 plateau → 严重 undershoot。须 min_sweeps≥~T_MAX 再允许 plateau 判据生效。
4. **warm-start fallback（sweep1 MSE>10×warm末）在大网上过敏**：大网 warm 拟合太好→warm末 MSE 极小→10× 阈太低→误触发冷启。但 FVI 对 init 不敏感，冷启照样收敛（PASS 不受影响）。下次可改绝对阈或更大倍数。
5. **每 head 单独跑会覆盖 JSON** → main 读已有 JSON merge 再写。

## 不认证项（声明，类 R3 F3）
4×4 单块/回合 + 线性计分无 combo。双 gate **不**认证 8×8 beam_hand 三块手牌候选枚举×afterstate 集成（靠 8×8 competence gate vs beam-strong paired-CRN + 预注册效应量兜底）。本认证只说"FVI 机器在可精确求解小盘上复现真·最优 + 真·有限-T 值"。

见 [[rl_plan]]（设计 + 三轮审核）、[[oracle_immortality_reframe]]（六大陷阱）、log.md 续9。
