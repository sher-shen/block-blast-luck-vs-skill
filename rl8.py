"""rl8.py — 8×8 afterstate-FVI harness（torch；与 rl4.py 并列，仅此二文件依赖 torch）。

定位（PLAN_8x8_RL.md §3）：把 4×4 认证过的 FVI 思路搬到 8×8。8×8 reachable 不可枚举
→ 训练态必须**采样**（self-play rollout buffer），CNN 在采样到的 afterstate 上半梯度回归。
架构(CNN)、状态采样(buffer)、损失(per-round-rate 变换)三者全换 → 必须先过 §3.0 迁移 gate
（同一份网络/forward/编码代码，在 4×4 引擎上重训权重，复现 backward_dp_T(8/16)）才放行 8×8。

预注册设计决策（§3.1 内部张力的定调，loop 2026-06-01）：
  afterstate-V 条件于 **(board, combo, k)**，**不含手牌 glyph 平面**。理由：§3.3 的权威
  Bellman 目标 y(s,k)=mean_p max(cl+V(s',k-1)) 中手牌已被消费、由 mean_p 平均掉；与 rl4
  mode-T（board+k 无手牌）+ fast.beam_hand 返回 (board,combo,score) 的 afterstate 粒度一致。
  §3.1 首条 bullet 的"3 手牌平面"属 pre-placement 状态编码，afterstate-V 不取。

per-round-rate 变换（§3.6 复审定稿，反 Huber-on-raw）：网络头输出 per-round-rate
rate_net(s,k)；Bellman 在**总值**域递归，严格三步：
  (1) 还原总值 V_total(s',k-1) = rate_net(s',k-1) × (k-1)（k-1==0 → 0）；
  (2) y_total(s,k) = mean_p max_pos(cl + V_total(s',k-1))；
  (3) 回归目标 y_rate(s,k) = y_total(s,k) / k，对 rate_net(s,k) 做 MSE。
下游 competence/EVPI 用 V 一律先 ×k 还原成总值。**绝不**把 rate 直接塞进 max。

硬约定（PLAN §0）：print 全 flush=True；目标里不含 heuristic（warm-start 只进 init）；
mode-T 无折扣有限-horizon（V 条件于 rounds-left k），匹配 channelB 固定-T 无折扣求和。
"""

import json
import random
import sys
import time

import torch
import torch.nn as nn

torch.manual_seed(0)
# §3.2：先 CPU；实测 CNN 全 reachable(41503) full-batch FVI 在 CPU=2187ms/fwd+bwd
# → 迁移 gate ~180s/sweep（5h+ 才收敛）。MPS=211ms（10×）→ ~17s/sweep。故默认 MPS。
DEVICE = ("cuda" if torch.cuda.is_available()
          else "mps" if torch.backends.mps.is_available()
          else "cpu")  # Windows+NVIDIA → cuda; Mac → mps; 否则 cpu
BOARD_DIM = 8           # 网络永远吃 8×8 平面（4×4 引擎 zero-pad 进左上角）
T_MAX = 16              # 迁移 gate 在 k∈{8,16}；8×8 训练时按 HORIZON_T 调
# 补20：无折扣 mode-T 在 T=50 长程发散（deadly triad 跨 50 层复利，logv/rate 都爆）。
# GAMMA<1 当**训练稳定器**（γ-收缩保证有界收敛，4×4 dp4 已验同款）。默认 1.0=无折扣（保 4×4 gate）；
# 8×8 训练设 0.95。**strength-first：γ 只塑造策略，判强/天花板用无折扣实战得分(rollout)，γ 不进 channelB → §0.2 不破。**
GAMMA = 1.0

# 迁移 gate 预注册真值（dp4 现码现跑，与 rl4.py 同源）
BDP_T = {8: 3.5934, 16: 4.8637}


# ====================================================================
# 编码：board int → 8×8 平面（+ combo, k 标量在 forward 里 concat）
# ====================================================================
def board_plane(board, side):
    """board int（side×side bitboard，bit r*side+c）→ 8×8 float plane（CPU 构建，左上角 side×side）。
    pad 区 = 0（空格，合法空盘格，无伪影特判，§3.0）。返回 (1,8,8) CPU tensor。
    注意**在 CPU 构建**：逐格赋值在 MPS 上是上千次慢分发，构建后由 batch 一次性搬 DEVICE。"""
    plane = torch.zeros(BOARD_DIM, BOARD_DIM)
    for i in range(side * side):
        if (board >> i) & 1:
            r, c = divmod(i, side)
            plane[r, c] = 1.0
    return plane.unsqueeze(0)


def net_forward_chunked(net, planes, combo, kn, chunk=120000):
    """分块 forward（防 MPS OOM）：eval-dst 集可达 ~1e6 boards，单批 conv 激活
    (N,32,8,8) 会爆显存（补16：buffer 长大后 11.5GiB OOM）。分块把激活峰值压到 chunk×32×8×8。"""
    if planes.shape[0] <= chunk:
        return net(planes, combo, kn)
    outs = []
    for i in range(0, planes.shape[0], chunk):
        outs.append(net(planes[i:i + chunk], combo[i:i + chunk], kn[i:i + chunk]))
    return torch.cat(outs, dim=0)


def board_planes_batch(boards, side):
    """一批 board int → (Nb,1,8,8)，CPU 构建后一次性搬到 DEVICE。"""
    return torch.stack([board_plane(b, side) for b in boards], dim=0).to(DEVICE)


# ====================================================================
# 网络：小 CNN，输出 per-round-rate 标量（§3.2 / §3.6）
# ====================================================================
class RateNet(nn.Module):
    """2 conv（padding='same'，全卷积保空间维）→ flatten → concat(combo,k) → MLP → rate。
    ~140k 参数，CPU。迁移 gate 与 8×8 共用此类，仅重训权重。"""

    def __init__(self, ch1=32, ch2=32, hidden=64):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, ch1, 3, padding=1), nn.ReLU(),
            nn.Conv2d(ch1, ch2, 3, padding=1), nn.ReLU(),
        )
        self.flat_dim = ch2 * BOARD_DIM * BOARD_DIM
        self.head = nn.Sequential(
            nn.Linear(self.flat_dim + 2, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, planes, combo, k_norm):
        """planes (Nb,1,8,8); combo (Nb,) 原始 combo 计数; k_norm (Nb,) 归一化 rounds-left。
        combo 归一化 /10 保 O(1)（与 conv 特征同量级；combo=0→0，4×4 gate 不受影响）。→ rate。"""
        z = self.conv(planes).flatten(1)
        scal = torch.stack([combo / 10.0, k_norm], dim=1)
        return self.head(torch.cat([z, scal], dim=1)).squeeze(-1)


# ====================================================================
# 引擎抽象：4×4(dp4) 与 8×8(fast) 提供各自的 round 动力学
# ====================================================================
class Engine4:
    """4×4 dp4 引擎：单块/回合、线性计分（无 combo）。reachable 可枚举（仅供 gate 覆盖参照）。"""
    side = 4
    has_combo = False
    train_on_closure = True      # gate：训全 reachable 闭包（续值全覆盖，验架构不被覆盖率混淆）

    def __init__(self):
        import dp4
        self.dp4 = dp4
        self.K = dp4.K
        self.PLACE = dp4.PLACE4
        self.apply = dp4.apply4
        self.legal_moves = dp4.legal_moves

    def sample_hand(self, rng):
        """4×4 一回合发 1 块。"""
        return [rng.randrange(self.K)]

    def round_candidates(self, board, combo, hand):
        """放完整手后的 afterstate 候选 [(board', combo', hand_score)]。
        4×4 单块：枚举该块所有合法落点，hand_score=cleared。"""
        pid = hand[0]
        out = []
        for m in self.PLACE[pid]:
            if board & m:
                continue
            nb, cl = self.apply(board, m)
            out.append((nb, 0, float(cl)))
        return out

    def enumerate_hands(self):
        """所有可能手（4×4=每个 piece 一手），用于 mean_p 精确平均。返回 [(hand, prob)]。"""
        return [([p], 1.0 / self.K) for p in range(self.K)]


class Engine8:
    """8×8 fast 引擎：3 块手牌、combo 计分。afterstate 由 beam_hand 给候选。"""
    side = 8
    has_combo = True
    train_on_closure = False     # 8×8 不可枚举：训采样 buffer，覆盖率由 BUFFER_COVERAGE_τ 单独 gate

    def __init__(self, beam_B=12):
        import fast
        self.fast = fast
        self.NUM_TYPES = fast.NUM_TYPES
        self.beam_B = beam_B

    def sample_hand(self, rng):
        return [rng.randrange(self.NUM_TYPES) for _ in range(3)]

    def round_candidates(self, board, combo, hand):
        """beam_hand top-B afterstate 候选 [(board', combo', hand_score)]。"""
        return self.fast.beam_hand(board, combo, hand, self.beam_B)


# ====================================================================
# 采样 buffer（§3.4）：self-play（+ε）收集途经 afterstate board
# ====================================================================
def collect_buffer(engine, net, n_rollouts, T, eps, seed_tag, extra_policies=None, transform="rate"):
    """从空盘用当前 rate_net 贪心(+ε) 跑 self-play，收集途经 afterstate **(board, combo)**（去重）。
    combo 跨轮串联（real threading）：8×8 afterstate-V 条件于 (board,combo,k)，combo 必须随状态走。
    extra_policies: 可注入 off-policy rollout 做 strong/seer 补采（§3.4 分布漂移防护）。
    返回 set((board:int, combo:int))。"""
    seen = {(0, 0)}
    for r in range(n_rollouts):
        rng = random.Random(f"{seed_tag}-{r}")
        board, combo = 0, 0
        for i in range(T):
            k = T - i
            hand = engine.sample_hand(rng)
            cands = engine.round_candidates(board, combo, hand)
            if not cands:
                break
            if rng.random() < eps:
                nb, nc, _ = cands[rng.randrange(len(cands))]
            else:
                nb, nc = _greedy_pick(net, engine, cands, k, transform)
            board, combo = nb, nc
            seen.add((board, combo))
    return seen


def forward_closure(engine, seed_boards, hand_list, max_boards=60000):
    """从 seed_boards 出发，沿所有手的 afterstate 反复扩张到不动点（或撞 max_boards）。
    4×4：closure({0}) = 全 reachable（41503<cap）→ 续值查询全覆盖，gate 验架构不被覆盖率混淆。
    8×8：远超 cap → 截断成 buffer+frontier（覆盖率由 BUFFER_COVERAGE_τ 单独把关，§3.4）。"""
    seen = set(seed_boards)
    frontier = list(seen)
    hands = [h for h, _ in hand_list]
    while frontier and len(seen) < max_boards:
        nxt = []
        for b in frontier:
            for hand in hands:
                for (s2, _c, _hs) in engine.round_candidates(b, 0, hand):
                    if s2 not in seen:
                        seen.add(s2); nxt.append(s2)
                        if len(seen) >= max_boards:
                            break
                if len(seen) >= max_boards:
                    break
            if len(seen) >= max_boards:
                break
        frontier = nxt
    return sorted(seen)


def exact_vtot_anchor(engine, hands, T):
    """8×8 **精确真值锚**（复审荐）：用训练同一组固定 hands 做精确后向归纳（非 MC），
    递归记忆 (board,combo,k)。V_total(s,k)=Σ_h prob·max_c(hs+V(s',k-1))，V(·,0)=0。
    返回 {k: V_total(empty,k)} for k=1..T。T≤3 可秒级精算（8 hands×~12 cand 分支 + memo）。
    这是 pipeline(固定-hand 动力学) 的真值 → 第一个真正能在 8×8 重尾域验变换正确性的靶。"""
    memo = {}
    def V(board, combo, k):
        if k == 0:
            return 0.0
        key = (board, combo, k)
        if key in memo:
            return memo[key]
        tot = 0.0
        for hand, prob in hands:
            best = 0.0
            for (s2, c2, hs) in engine.round_candidates(board, combo, hand):
                v = hs + V(s2, c2, k - 1)
                if v > best:
                    best = v
            tot += prob * best
        memo[key] = tot
        return tot
    return {k: V(0, 0, k) for k in range(1, T + 1)}


def _greedy_pick(net, engine, cands, k, transform="rate"):
    """对候选 argmax hand_score + V_total(board', k-1)。返回 (board', combo')。"""
    if k - 1 == 0:
        best = max(cands, key=lambda c: c[2])
        return best[0], best[1]
    boards = [c[0] for c in cands]
    combos = [float(c[1]) for c in cands]
    with torch.no_grad():
        planes = board_planes_batch(boards, engine.side)
        combo_t = torch.tensor(combos, device=DEVICE)
        kn = torch.full((len(cands),), (k - 1) / T_MAX, device=DEVICE)
        vtot = decode_vtot(net(planes, combo_t, kn), k - 1, transform)
    scores = [c[2] + GAMMA * float(vtot[j]) for j, c in enumerate(cands)]   # 补20：γ 一致
    j = max(range(len(cands)), key=lambda x: scores[x])
    return cands[j][0], cands[j][1]


# ====================================================================
# 转移表预计算（rl4.build_transitions 的 buffer 版）：一次性枚举 src×hand×候选，
# 之后每 sweep 只做"net 评 dst 续值 + 向量化 scatter-amax"。dst 不必在 buffer 里
# （CNN 泛化），dst 去重成 eval 集供批量 forward。
# ====================================================================
def build_buffer_transitions(engine, buf, hand_list):
    """buf = [(board, combo)]（afterstate）。返回向量化转移表。
    group = src_idx*nhands + hand_idx，scatter-amax 段；段 init=0 → max(0,·) 与放不下→0。
    eval 集按 **(dst_board, dst_combo)** 去重（combo 是状态量，影响 V）。src/eval 都带真实 combo。"""
    nsrc = len(buf)
    nhands = len(hand_list)
    probs = [p for _, p in hand_list]
    dst_set = {}                       # (dst_board, dst_combo) → eval idx
    dst_idx, hs_list, grp = [], [], []
    for si, (src_board, src_combo) in enumerate(buf):
        for hi, (hand, _p) in enumerate(hand_list):
            for (s2, c2, hs) in engine.round_candidates(src_board, src_combo, hand):
                key = (s2, c2)
                if key not in dst_set:
                    dst_set[key] = len(dst_set)
                dst_idx.append(dst_set[key])
                hs_list.append(hs); grp.append(si * nhands + hi)
    # eval 集按插入序（= dst_set 的 idx）排列，与 dst_idx 对齐
    eval_keys = [k for k, _ in sorted(dst_set.items(), key=lambda kv: kv[1])]
    eval_boards = [b for b, _ in eval_keys]
    eval_combos = [float(c) for _, c in eval_keys]
    src_planes = board_planes_batch([b for b, _ in buf], engine.side)
    src_combo = torch.tensor([float(c) for _, c in buf], device=DEVICE)
    eval_planes = board_planes_batch(eval_boards, engine.side)
    return {
        "src_planes": src_planes,
        "src_combo": src_combo,
        "eval_planes": eval_planes,
        "eval_combo": torch.tensor(eval_combos, device=DEVICE),
        "dst_idx": torch.tensor(dst_idx, dtype=torch.long, device=DEVICE),
        "hand_score": torch.tensor(hs_list, device=DEVICE),
        "group": torch.tensor(grp, dtype=torch.long, device=DEVICE),
        "hand_prob": torch.tensor(probs, device=DEVICE),
        "nsrc": nsrc, "nhands": nhands,
        "n_eval": eval_planes.shape[0],
    }


# ---------- §3.6 变换：net 输出空间 ↔ V_total（总值域），用于 micro-check 对比 ----------
# 网络输出 u；decode_vtot 还原 V_total 供 Bellman max；encode_target 把 y_total 映回 u 空间作回归目标。
# 'rate'：u=rate=V/k（plan 默认）；'lograte'：u=log1p(V/k)（压重尾）；loss 在 train_modeT 选。
def decode_vtot(u, k, transform):
    """net 输出 u（在 rounds-left k）→ V_total。k==0 调用方已短路为 0。"""
    if transform == "lograte":
        return torch.expm1(u) * k          # u=log1p(V/k) → V=expm1(u)*k（高 k ×k 放大 log 误差，复审证 FAIL）
    if transform == "logv":
        # u=log1p(V) 直压总值，**无 ÷k/×k**（复审荐：去 ×k 放大器）。
        # 补17：clamp(max=11) → V_total ≤ expm1(11)≈5.9e4 物理天花板（seer T=50 ~2.7e4 < 此，不偏真值），
        # 硬断 expm1 在 max-自举正反馈下的 runaway→NaN（logv 发散根因）。
        return torch.expm1(torch.clamp(u, max=11.0))
    # 'rate' / 'rate_huber'：u=per-round-rate，V=u×k（线性，无 expm1 爆炸）。
    # 补18：clamp(max=5000) 纯防 inf 保险（真实 per-round-rate ~O(数百)，5000 远不 bind、不偏置；
    # 4×4 rate~O(1) 不受影响）。strength-first 选 rate：线性稳收敛，×k 偏置记目标②局限。
    return torch.clamp(u, max=5000.0) * k

def encode_target(y_total, k, transform):
    """y_total（总值域目标）→ net 输出空间的回归目标 u_target。"""
    if transform == "lograte":
        return torch.log1p(y_total / k)
    if transform == "logv":
        return torch.log1p(y_total)        # 直接 log 压总值，k 依赖由网络的 k 输入学
    return y_total / k


def y_total_from_trans(net, tr, k, transform="rate"):
    """用预计算转移表 + 当前 net 算 y_total(src,k)（精确 mean_p）。
    V_total(dst,k-1)=decode_vtot(net(dst,k-1), k-1)；k-1==0→0。返回 (nsrc,)。"""
    nsrc, nhands = tr["nsrc"], tr["nhands"]
    if k - 1 == 0:
        vtot_eval = torch.zeros(tr["n_eval"], device=DEVICE)
    else:
        with torch.no_grad():
            kn = torch.full((tr["n_eval"],), (k - 1) / T_MAX, device=DEVICE)
            vtot_eval = decode_vtot(
                net_forward_chunked(net, tr["eval_planes"], tr["dst_combo_eval"], kn),
                k - 1, transform)
    cand_val = tr["hand_score"] + GAMMA * vtot_eval[tr["dst_idx"]]   # 补20：γ-折扣续值
    seg = torch.zeros(nsrc * nhands, device=DEVICE)
    seg = seg.scatter_reduce(0, tr["group"], cand_val, reduce="amax", include_self=True)
    per_hand = seg.view(nsrc, nhands)                         # max_pos per (src,hand)
    return (per_hand * tr["hand_prob"]).sum(dim=1)            # mean_p


def train_modeT(engine, max_sweeps=400, inner_steps=2, lr=1e-3, hidden=64,
                plateau_tol=0.005, plateau_K=8, min_sweeps=80, wall_budget_s=1800,
                n_rollouts=400, eps=0.3, hand_samples=8, recollect_every=20,
                ckpt_path=None, resume_ckpt=None, ckpt_every=20, transform="rate",
                buf_override=None, offpolicy=None, grad_clip=None):
    """无折扣有限-horizon mode-T FVI。self-play buffer + 预计算转移表。
    transform（§3.6 micro-check）：'rate'=per-round-rate+MSE(默认) / 'rate_huber'=rate+Huber /
    'lograte'=log1p(rate)+MSE（压 combo 重尾）。**冻结每-sweep 目标**（rl4 式，避免 k 轴
    interference）。ckpt_path/resume_ckpt：断点续训。返回 dict。"""
    t0 = time.time()
    net = RateNet(hidden=hidden).to(DEVICE)
    if resume_ckpt:
        try:
            net.load_state_dict(torch.load(resume_ckpt, map_location=DEVICE))
            print(f"[mode-T/{engine.side}] resumed net from {resume_ckpt}", flush=True)
        except FileNotFoundError:
            print(f"[mode-T/{engine.side}] resume_ckpt {resume_ckpt} 不存在 → 冷启", flush=True)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    loss_fn = nn.HuberLoss(delta=1.0) if transform == "rate_huber" else nn.MSELoss()
    exact_hands = engine.enumerate_hands() if hasattr(engine, "enumerate_hands") else None
    if exact_hands is None:
        # 8×8：采样手（固定种子，整轮训练复用同一组以稳定目标）
        rng = random.Random("rl8-hands")
        exact_hands = [(engine.sample_hand(rng), 1.0 / hand_samples)
                       for _ in range(hand_samples)]

    # §3.4 off-policy 补采（8×8）：strong+seer rollout 态混入 buffer，覆盖 self-play 外高-combo 尾部。
    # 一次性算（贵），整训复用 → 让自洽检验测【变换】而非【覆盖】。offpolicy=(n_strong,n_seer)。
    offpolicy_set = set()
    if offpolicy is not None and getattr(engine, "side", 4) == 8:
        ns, nz = offpolicy
        print(f"[mode-T/8] off-policy 补采 strong×{ns} + seer×{nz} rollout 态…", flush=True)
        offpolicy_set = offpolicy_states(engine, n_strong=ns, n_seer=nz, seed="rl8-off")
        print(f"[mode-T/8] off-policy 补采 |states|={len(offpolicy_set)}", flush=True)

    def collect():
        sp = collect_buffer(engine, net, n_rollouts, T_MAX, eps, "rl8-buf", transform=transform)
        return sorted(sp | offpolicy_set)

    # self-play 采样 buffer（§3.4）。4×4 gate：训练集=前向闭包（全 reachable→续值全覆盖，
    # 验架构不被覆盖率混淆）。8×8：不可枚举→训练集=采样 buffer 本身，覆盖率由下方 dst-in-buf
    # 诊断 + Phase 3 的 BUFFER_COVERAGE_τ 把关。
    def make_buf(buf_sampled):
        """buf_sampled=set((board,combo))。4×4 gate→训前向闭包(全 reachable,combo≡0)；
        8×8→训采样集本身(tuples)。返回 (buf=[(board,combo)], coverage)。"""
        if getattr(engine, "train_on_closure", True):
            seed_boards = {b for b, _ in buf_sampled}
            closure = forward_closure(engine, seed_boards, exact_hands)
            buf = [(b, 0) for b in closure]                  # 4×4 combo≡0
            return buf, (len(buf_sampled) / len(buf) if buf else 0.0)
        return sorted(buf_sampled), 1.0                       # 8×8 训练集=采样集

    if buf_override is not None:
        buf_sampled = set(buf_override)               # 隔离实验：注入满覆盖训练集（绕过 self-play）
        buf, coverage = sorted(buf_override), 1.0
        recollect_every = 10**9
    else:
        buf_sampled = set(collect()) | {(0, 0)}
        buf, coverage = make_buf(buf_sampled)
    tr = build_buffer_transitions(engine, buf, exact_hands)
    tr["dst_combo_eval"] = tr["eval_combo"]
    print(f"[mode-T/{engine.side}x{engine.side}] sampled |buf|={len(buf_sampled)}  "
          f"train |buf|={len(buf)}  cover={coverage:.2%}  |eval-dst|={tr['n_eval']}  "
          f"trans={tr['group'].numel()}  rollouts={n_rollouts} eps={eps}", flush=True)

    combo_buf = tr["src_combo"]
    probe_hist, converged, sweeps_done = [], False, 0
    for sweep in range(max_sweeps):
        if sweep > 0 and sweep % recollect_every == 0:
            buf_sampled |= set(collect())
            buf, coverage = make_buf(buf_sampled)
            tr = build_buffer_transitions(engine, buf, exact_hands)
            tr["dst_combo_eval"] = tr["eval_combo"]
            combo_buf = tr["src_combo"]
        # 1) 用冻结 net 建全 k 目标（§3.6：encode_target 把 y_total 映回 net 输出空间），训练时不动目标
        frozen = {}
        for k in range(1, T_MAX + 1):
            frozen[k] = encode_target(y_total_from_trans(net, tr, k, transform), k, transform).detach()
        kn_of = {k: torch.full((tr["nsrc"],), k / T_MAX, device=DEVICE)
                 for k in range(1, T_MAX + 1)}
        # 2) 对冻结目标训 inner_steps 遍（每遍过全 k），共同拟合避免 k 轴 interference
        sweep_loss = 0.0
        for _ in range(inner_steps):
            for k in range(1, T_MAX + 1):
                opt.zero_grad()
                loss = loss_fn(net(tr["src_planes"], combo_buf, kn_of[k]), frozen[k])
                loss.backward()
                if grad_clip is not None:
                    nn.utils.clip_grad_norm_(net.parameters(), grad_clip)  # 补17：防 value blow-up
                opt.step()
                sweep_loss += loss.item()
        with torch.no_grad():
            pe = board_planes_batch([0], engine.side)
            z0 = torch.zeros(1, device=DEVICE)
            v8 = float(decode_vtot(net(pe, z0, torch.full((1,), 8 / T_MAX, device=DEVICE)), 8, transform)[0])
            v16 = float(decode_vtot(net(pe, z0, torch.full((1,), 16 / T_MAX, device=DEVICE)), 16, transform)[0])
        probe_hist.append(torch.tensor([v8, v16]))
        sweeps_done = sweep + 1
        if sweep % 10 == 0 or sweep < 3:
            print(f"[mode-T/{engine.side}] sweep {sweep:3d}  Vtot(0,8)={v8:.4f}  "
                  f"Vtot(0,16)={v16:.4f}  loss={sweep_loss:.5f}  |buf|={len(buf)}", flush=True)
        if ckpt_path and sweep > 0 and sweep % ckpt_every == 0:
            torch.save(net.state_dict(), ckpt_path)
        if sweep + 1 >= min_sweeps and len(probe_hist) > plateau_K:
            recent = torch.stack(probe_hist[-(plateau_K + 1):])
            rel = (recent[1:] - recent[:-1]).abs() / (recent[:-1].abs() + 1e-6)
            if rel.max().item() < plateau_tol:
                converged = True
                print(f"[mode-T/{engine.side}] PLATEAU @ sweep {sweep} "
                      f"(max rel {rel.max():.4%})", flush=True)
                break
        if time.time() - t0 > wall_budget_s:
            print(f"[mode-T/{engine.side}] WALL-CLOCK 超预算 {wall_budget_s}s → 停", flush=True)
            break

    wall = time.time() - t0
    if ckpt_path:
        torch.save(net.state_dict(), ckpt_path)
    with torch.no_grad():
        pe = board_planes_batch([0], engine.side)
        z0 = torch.zeros(1, device=DEVICE)
        v8 = float(decode_vtot(net(pe, z0, torch.full((1,), 8 / T_MAX, device=DEVICE)), 8, transform)[0])
        v16 = float(decode_vtot(net(pe, z0, torch.full((1,), 16 / T_MAX, device=DEVICE)), 16, transform)[0])
    print(f"[mode-T/{engine.side}] DONE  Vtot(0,8)={v8:.4f}  Vtot(0,16)={v16:.4f}  "
          f"sweeps={sweeps_done}  wall={wall:.1f}s  plateau={converged}  "
          f"|buf|={len(buf)}", flush=True)
    return {"net": net, "engine": engine, "v8": v8, "v16": v16,
            "buf_size": len(buf), "buf_sampled": len(buf_sampled), "coverage": coverage,
            "sweeps": sweeps_done, "wall_s": wall, "plateau": converged,
            "wall_budget_exceeded": wall > wall_budget_s}


def train_onpolicy(transform="rate", T=50, n_outer=6, sweeps=30, cap=22000,
                   hidden=128, seed_ckpt=None, ckpt_out=None):
    """近似策略迭代：修 greedy-on-V 的分布漂移（competence 诊断出 rl 在 OOD 态过度高估→早死）。
    交替 (用当前 V 的贪心策略收集途经态) ↔ (在 含这些态+strong/seer 锚点 的 buffer 上重训 V)，
    让 V 在策略**真正会遇到**的局面上变准。可从 seed_ckpt（已 plateau 的 g98）热启。
    返回 {ckpt, hist}。"""
    eng = Engine8()
    ckpt = ckpt_out or f"/tmp/rl8_8x8_{transform}_T{T}_pi.pt"
    net = RateNet(hidden=hidden).to(DEVICE)
    if seed_ckpt:
        net.load_state_dict(torch.load(seed_ckpt, map_location=DEVICE))
        torch.save(net.state_dict(), ckpt)
        print(f"[PI] 热启 seed={seed_ckpt} → {ckpt}", flush=True)
    # strong+seer 锚点（始终保留，给 V 高质量覆盖）
    offpol = offpolicy_states(eng, n_strong=20, n_seer=20, seed="rl8-pi-off")
    onpol = collect_buffer(eng, net, 200, T, 0.1, "rl8-pi-0", transform=transform)
    buf = list(offpol | onpol | {(0, 0)})
    hist = []
    for it in range(n_outer):
        print(f"\n[PI outer {it}] |buf|={len(buf)}（onpol+anchor）→ 训 {sweeps} sweeps", flush=True)
        res = train_modeT(eng, transform=transform, hidden=hidden, lr=5e-4, inner_steps=3,
                          ckpt_path=ckpt, resume_ckpt=ckpt, ckpt_every=10,
                          max_sweeps=sweeps, min_sweeps=sweeps, wall_budget_s=3600,
                          grad_clip=1.0, buf_override=buf, recollect_every=10**9)
        net = RateNet(hidden=hidden).to(DEVICE)
        net.load_state_dict(torch.load(ckpt, map_location=DEVICE))
        new_on = collect_buffer(eng, net, 200, T, 0.1, f"rl8-pi-{it+1}", transform=transform)
        merged = set(buf) | new_on
        if len(merged) > cap:                      # 限容：保锚点 + 随机保留其余到 cap
            keep = set(offpol)
            others = [s for s in merged if s not in keep]
            rng = random.Random(f"pi-cap-{it}")
            rng.shuffle(others)
            merged = keep | set(others[:max(0, cap - len(keep))])
        buf = list(merged)
        hist.append({"outer": it, "v8": res["v8"], "v16": res["v16"], "buf": len(buf)})
        print(f"[PI outer {it}] v8={res['v8']:.1f} v16={res['v16']:.1f} → ckpt={ckpt}", flush=True)
    print(f"\n-> PI 完成 ckpt={ckpt}", flush=True)
    return {"ckpt": ckpt, "hist": hist}


# ====================================================================
# §3.4 off-policy 状态生成：strong + seer rollout 途经态（补采，防 on-policy 漂移）
# 用于 (a) 8×8 训练 buffer 补采；(b) §3.6★(c) 自洽探针采样。两处共用同一分布 →
# 自洽检验测的是【变换是否被 ×k 放大】而非【训练 buffer 是否覆盖探针】。
# ====================================================================
def _strong_states(engine, rng, n_rounds):
    """strong（手内 beam）真实玩 n_rounds，沿途收集 afterstate (board,combo)。"""
    out, board, combo = [], 0, 0
    for _ in range(n_rounds):
        hand = engine.sample_hand(rng)
        sc, nb, nc, alive = engine.fast.strong_hand(board, combo, hand, B=engine.beam_B)
        if not alive:
            break
        board, combo = nb, nc
        out.append((board, combo))
    return out


def _seer_states(engine, rng, n_rounds, look=4):
    """seer（**真未来**短-strong-rollout 前瞻）玩 n_rounds：每轮在 beam 候选里选
    hand_score + strong-rollout(真实剩余牌, look 手) 最大者 → 触及高-combo 尾部态。沿途收集 afterstate。
    这是 §3.6★ 想测 ×k 放大最该覆盖的态（重尾 = ×k 放大最猛处）。"""
    future = [engine.sample_hand(rng) for _ in range(n_rounds + look)]
    out, board, combo = [], 0, 0
    for i in range(n_rounds):
        cands = engine.round_candidates(board, combo, future[i])
        if not cands:
            break
        rem = [p for h in future[i + 1:i + 1 + look] for p in h]
        best = None
        for (s2, c2, hs) in cands:
            roll = engine.fast._rollout_fixed_strong(s2, c2, rem, B=6) if rem else 0.0
            val = hs + roll
            if best is None or val > best[0]:
                best = (val, s2, c2)
        board, combo = best[1], best[2]
        out.append((board, combo))
    return out


def offpolicy_states(engine, n_strong=20, n_seer=20, n_rounds=60, seed="off"):
    """汇集 strong + seer rollout 途经 (board,combo) 态（去重）。8×8 专用（combo 重尾）。"""
    seen = set()
    for r in range(n_strong):
        seen.update(_strong_states(engine, random.Random(f"{seed}-s-{r}"), n_rounds))
    for r in range(n_seer):
        seen.update(_seer_states(engine, random.Random(f"{seed}-z-{r}"), n_rounds))
    return seen


# ====================================================================
# §3.6★(c) 高-k 自洽真裁判：MC-rollout（无偏参照，不经 decode ×k）vs V_net
# ====================================================================
def board_planes_fast8(boards):
    """8×8 专用向量化 bit-unpack（split 32-bit 半避免 signed-int64 bit63 溢出）。
    boards: list[int]（8×8 bitboard，bit r*8+c）→ (Nb,1,8,8) DEVICE tensor。
    比逐格 python 赋值快 ~100×；rollout 百万级 board 编码必须用它（实测 10.7ms/24k boards）。"""
    low = [b & 0xFFFFFFFF for b in boards]
    high = [(b >> 32) & 0xFFFFFFFF for b in boards]
    lt = torch.tensor(low, dtype=torch.long)
    ht = torch.tensor(high, dtype=torch.long)
    ar = torch.arange(32)
    bits = torch.cat([(lt.unsqueeze(1) >> ar) & 1, (ht.unsqueeze(1) >> ar) & 1], dim=1)
    return bits.view(len(boards), 1, 8, 8).float().to(DEVICE)


def collect_probes(engine, n=32, seed="probe"):
    """§3.6★(c) 探针 = empty + n 个冻结中局 (board,combo)，从 seer/strong rollout 快照采
    （off-policy，覆盖 self-play 外高-combo 尾部）。冻结 → 跨 transform 配对可比。返回 [(board,combo)]。"""
    snaps = []
    nz = (n + 1) // 2
    for r in range((nz + 5) // 6):                       # seer 局（重尾尾部）
        snaps += _seer_states(engine, random.Random(f"{seed}-seer-{r}"), 60)[5::10]
    for r in range((n - nz + 5) // 6):                   # strong 局
        snaps += _strong_states(engine, random.Random(f"{seed}-str-{r}"), 60)[5::10]
    # 去重 + 冻结取前 n（种子固定 → 确定性）
    uniq = list(dict.fromkeys(snaps))
    return [(0, 0)] + uniq[:n]


def mc_rollout_value(net, engine, probes, k, M, transform, seed_base="sc"):
    """§3.6★(c) 无偏参照：每 probe (board,combo) 从该态用**贪心-on-V_net 真实玩 k 轮、
    累加真实 hand_score**，M 条独立未来发牌。**返回值是真实累加分，全程不经 decode ×k**
    （decode 只在 argmax 选点用）→ 对 ×k 放大免疫的 V^π(s,k) 估计。
    向量化：M episodes 并行，每轮 beam_hand 在 python（瓶颈~1.7ms/call），net 评分批量 1 次。
    返回 list[{mean, block_se, alive_frac}]（16-block-SE，§0.3）。"""
    out = []
    for p_idx, (pb, pc) in enumerate(probes):
        boards = [pb] * M; combos = [pc] * M; totals = [0.0] * M; alive = [True] * M
        rngs = [random.Random(f"{seed_base}-{p_idx}-{m}") for m in range(M)]
        for rnd in range(k):
            kleft = k - rnd
            flat_b, flat_c, flat_hs, span = [], [], [], {}
            for m in range(M):
                if not alive[m]:
                    continue
                cands = engine.round_candidates(boards[m], combos[m], engine.sample_hand(rngs[m]))
                if not cands:
                    alive[m] = False; continue
                start = len(flat_b)
                for (s2, c2, hs) in cands:
                    flat_b.append(s2); flat_c.append(float(c2)); flat_hs.append(hs)
                span[m] = (start, len(flat_b))
            if not flat_b:
                break
            if kleft - 1 == 0:
                vtot = [0.0] * len(flat_b)
            else:
                with torch.no_grad():
                    kn = torch.full((len(flat_b),), (kleft - 1) / T_MAX, device=DEVICE)
                    vtot = decode_vtot(net(board_planes_fast8(flat_b),
                                           torch.tensor(flat_c, device=DEVICE), kn),
                                       kleft - 1, transform).tolist()
            for m, (s, e) in span.items():
                # 策略选点用 γ-折扣续值（与训练一致）；但 totals 累加**原始** hand_score（无折扣实战得分=判强口径）
                bj, bv = s, flat_hs[s] + GAMMA * vtot[s]
                for j in range(s + 1, e):
                    v = flat_hs[j] + GAMMA * vtot[j]
                    if v > bv:
                        bv, bj = v, j
                boards[m] = flat_b[bj]; combos[m] = int(flat_c[bj]); totals[m] += flat_hs[bj]
        nb = 16; blk = M // nb
        bmeans = [sum(totals[i * blk:(i + 1) * blk]) / blk for i in range(nb)]
        mean = sum(bmeans) / nb
        se = (sum((x - mean) ** 2 for x in bmeans) / (nb - 1) / nb) ** 0.5
        out.append({"mean": mean, "block_se": se, "alive_frac": sum(alive) / M})
    return out


def vnet_predict(net, engine, probes, k, transform):
    """V_net(probe,k) decode 成总值（与 mc_rollout_value 对比）。"""
    boards = [b for b, _ in probes]; combos = [float(c) for _, c in probes]
    planes = board_planes_fast8(boards) if engine.side == 8 else board_planes_batch(boards, engine.side)
    with torch.no_grad():
        kn = torch.full((len(probes),), k / T_MAX, device=DEVICE)
        return decode_vtot(net(planes, torch.tensor(combos, device=DEVICE), kn), k, transform).tolist()


def selfcons_highk(transform, ckpt, hidden=128, M=2000, ks=(25, 50), n_probes=32,
                   out_json="rl8_selfcons.json"):
    """§3.6★(c) 高-k 自洽真裁判 orchestration（需先把 T_MAX 设为生产 horizon=50）。
    载 ckpt → 建 off-policy 探针 → 各 k 比 V_net vs MC-rollout → 判 PASS（rel@k50<0.10 且
    rel@k50/rel@k25<2.0 无-×k 放大签名）。写 JSON + 打印。返回 verdict dict。"""
    eng = Engine8()
    net = RateNet(hidden=hidden).to(DEVICE)
    net.load_state_dict(torch.load(ckpt, map_location=DEVICE))
    print(f"=== §3.6★(c) 高-k 自洽 | transform={transform} | T_MAX={T_MAX} | ckpt={ckpt} "
          f"| M={M} probes={n_probes}+empty ks={ks} ===", flush=True)
    probes = collect_probes(eng, n=n_probes)
    print(f"[selfcons] 探针 {len(probes)} 个（empty + {len(probes)-1} off-policy 中局态）", flush=True)
    res = {"transform": transform, "T_MAX": T_MAX, "M": M, "n_probes": len(probes), "ks": list(ks), "k": {}}
    for k in ks:
        t0 = time.time()
        roll = mc_rollout_value(net, eng, probes, k, M, transform)
        pred = vnet_predict(net, eng, probes, k, transform)
        rels = [abs(pred[i] - roll[i]["mean"]) / (abs(roll[i]["mean"]) + 1e-9) for i in range(len(probes))]
        srt = sorted(rels); med = srt[len(srt) // 2]
        res["k"][k] = {"median_rel": med,
                       "empty_pred": pred[0], "empty_roll": roll[0]["mean"], "empty_se": roll[0]["block_se"],
                       "probes": [{"pred": pred[i], "roll": roll[i]["mean"], "se": roll[i]["block_se"],
                                   "rel": rels[i], "alive": roll[i]["alive_frac"]} for i in range(len(probes))]}
        print(f"[selfcons] k={k}: median rel={med:.4f}  empty: V_net={pred[0]:.1f} "
              f"vs MC={roll[0]['mean']:.1f}±{roll[0]['block_se']:.1f}  ({time.time()-t0:.0f}s)", flush=True)
    r25, r50 = res["k"][ks[0]]["median_rel"], res["k"][ks[-1]]["median_rel"]
    ratio = r50 / (r25 + 1e-9)
    pass1 = r50 < 0.10
    pass2 = ratio < 2.0
    res["amplification_ratio"] = ratio
    res["pass_rel"] = bool(pass1); res["pass_no_amp"] = bool(pass2)
    res["PASS"] = bool(pass1 and pass2)
    print(f"\n[selfcons] === VERDICT transform={transform} ===", flush=True)
    print(f"  rel@k{ks[-1]}={r50:.4f} <0.10? {pass1}  |  放大签名 rel@k{ks[-1]}/k{ks[0]}={ratio:.2f} <2.0? {pass2}",
          flush=True)
    print(f"  → {'PASS（可定盘）' if res['PASS'] else 'FAIL'}", flush=True)
    try:
        allres = json.load(open(out_json))
    except (FileNotFoundError, json.JSONDecodeError):
        allres = {}
    allres[transform] = res
    json.dump(allres, open(out_json, "w"), ensure_ascii=False, indent=2)
    print(f"-> wrote {out_json}", flush=True)
    return res


# ====================================================================
# §5/§6 Phase 3 competence gate：rl-greedy-on-V vs strong（CRN 配对，固定 T）
# strength-first 真裁判 = 实战累加分（无折扣），不看 V 校准。
# ====================================================================
def _play_rl_streams(net, engine, streams, transform):
    """rl-greedy-on-V 在 M 条**固定发牌流**(CRN) 上各玩到死/到 T。向量化：M episodes 并行，
    每轮把所有 alive episode 的候选 afterstate 批量过 net 一次（与 mc_rollout_value 同骨架）。
    选点用 γ-折扣续值（与训练/rollout 一致）；totals 累加**原始** hand_score（无折扣判强口径）。
    返回 (totals:list[float], survived:list[int])。"""
    M = len(streams); T = len(streams[0])
    boards = [0] * M; combos = [0] * M; totals = [0.0] * M
    alive = [True] * M; survived = [0] * M
    for rnd in range(T):
        kleft = T - rnd
        flat_b, flat_c, flat_hs, span = [], [], [], {}
        for m in range(M):
            if not alive[m]:
                continue
            cands = engine.round_candidates(boards[m], combos[m], streams[m][rnd])
            if not cands:
                alive[m] = False; continue
            start = len(flat_b)
            for (s2, c2, hs) in cands:
                flat_b.append(s2); flat_c.append(float(c2)); flat_hs.append(hs)
            span[m] = (start, len(flat_b))
        if not flat_b:
            break
        if kleft - 1 == 0:
            vtot = [0.0] * len(flat_b)
        else:
            with torch.no_grad():
                kn = torch.full((len(flat_b),), (kleft - 1) / T_MAX, device=DEVICE)
                vtot = decode_vtot(net(board_planes_fast8(flat_b),
                                       torch.tensor(flat_c, device=DEVICE), kn),
                                   kleft - 1, transform).tolist()
        for m, (s, e) in span.items():
            bj, bv = s, flat_hs[s] + GAMMA * vtot[s]
            for j in range(s + 1, e):
                v = flat_hs[j] + GAMMA * vtot[j]
                if v > bv:
                    bv, bj = v, j
            boards[m] = flat_b[bj]; combos[m] = int(flat_c[bj]); totals[m] += flat_hs[bj]
            survived[m] = rnd + 1
    return totals, survived


def _play_baseline_stream(engine, stream, kind):
    """在一条固定发牌流上玩 strong 或 greedy（CPU，无前瞻）。返回 (total, survived_rounds)。"""
    fast = engine.fast
    board, combo, total = 0, 0, 0.0
    for rnd, hand in enumerate(stream):
        if kind == "strong":
            sc, board, combo, alive = fast.strong_hand(board, combo, hand, B=engine.beam_B)
        else:  # greedy（盲贪心，无 V、无前瞻）= 弱基线，验 rl 是否碾压启发式
            sc, board, combo, alive = fast.greedy_hand(board, combo, hand)
        if not alive:
            return total, rnd
        total += sc
    return total, len(stream)


def _paired_stats(diffs, nb=16, n_boot=2000, boot_seed="comp-boot"):
    """配对差值统计（§0.3 禁 ratio-of-means，纯差值）：mean + 16-block-SE +
    paired-bootstrap 90%/95% CI（确定性 seed）。diffs=逐 seed 的 (a−b)。"""
    M = len(diffs); mean = sum(diffs) / M
    blk = M // nb
    bmeans = [sum(diffs[i * blk:(i + 1) * blk]) / blk for i in range(nb)]
    bm = sum(bmeans) / nb
    block_se = (sum((x - bm) ** 2 for x in bmeans) / (nb - 1) / nb) ** 0.5
    rng = random.Random(boot_seed)
    boots = []
    for _ in range(n_boot):
        s = 0.0
        for _ in range(M):
            s += diffs[rng.randrange(M)]
        boots.append(s / M)
    boots.sort()
    def q(p):
        return boots[min(n_boot - 1, max(0, int(p * n_boot)))]
    return {"mean": mean, "block_se": block_se,
            "ci95": [q(0.025), q(0.975)], "ci90": [q(0.05), q(0.95)]}


def competence_gate(transform, ckpt, hidden=128, T=50, M_cal=300, M_eval=2000,
                    beam_B=12, out_json="rl8_competence.json"):
    """§5/§6 Phase 3：rl-greedy-on-V vs strong（CRN 配对，固定 T，无前瞻）= 最强策略 verdict。
    **预注册流程（§0.9）**：calibration seeds（与训练+eval 不相交）估 strong 均值 → 定死
    Δ_eq=0.05×strong_mean + 按 σ_diff 报 TOST 功效 → 再在 eval seeds 跑 → TOST 判 ≫/≈/≪。
    需先把 T_MAX=生产 horizon=50（CLI 已设）。beam_B=候选池大小（§5 disambiguation 放大到 64/200
    验 RL≪strong 是否同源小池假象）。返回 verdict dict。"""
    eng = Engine8(beam_B=beam_B)
    net = RateNet(hidden=hidden).to(DEVICE)
    net.load_state_dict(torch.load(ckpt, map_location=DEVICE))
    print(f"=== §5/§6 competence gate | transform={transform} T={T} | ckpt={ckpt} "
          f"| GAMMA={GAMMA}（选点折扣，得分无折扣累加） ===", flush=True)

    def gen_streams(tag, M):
        out = []
        for m in range(M):
            rng = random.Random(f"{tag}-{m}")
            out.append([eng.sample_hand(rng) for _ in range(T)])
        return out

    # ---- 1) CALIBRATION（与训练 seed["rl8-*"] + eval seed 不相交）：定 Δ_eq + 估 σ_diff ----
    t0 = time.time()
    cal_streams = gen_streams("comp-cal", M_cal)
    rl_cal, _ = _play_rl_streams(net, eng, cal_streams, transform)
    str_cal = [_play_baseline_stream(eng, s, "strong")[0] for s in cal_streams]
    strong_mean_cal = sum(str_cal) / M_cal
    DELTA_EQ = 0.05 * strong_mean_cal                       # 预注册 margin（单位=总分）
    diff_cal = [rl_cal[i] - str_cal[i] for i in range(M_cal)]
    dmean = sum(diff_cal) / M_cal
    sd = (sum((d - dmean) ** 2 for d in diff_cal) / (M_cal - 1)) ** 0.5
    se_eval = sd / (M_eval ** 0.5)
    # TOST 功效：true diff≈0 时 90% CI 半宽 = 1.645×SE_eval，须 < Δ_eq 才有等价检出力
    power_ok = (1.645 * se_eval) < DELTA_EQ
    print(f"[cal] M_cal={M_cal}  strong_mean={strong_mean_cal:.1f}  → Δ_eq=0.05×={DELTA_EQ:.2f}", flush=True)
    print(f"[cal] rl_mean={sum(rl_cal)/M_cal:.1f}  diff(rl-strong) mean={dmean:.1f}  sigma_diff={sd:.1f}", flush=True)
    print(f"[cal] M_eval={M_eval} → SE_eval≈{se_eval:.2f}; 1.645·SE={1.645*se_eval:.2f} "
          f"< Δ_eq={DELTA_EQ:.2f}? 等价检出力{'≥0.8 ✓' if power_ok else '不足 ⚠（CI 可能跨 Δ_eq → inconclusive）'}",
          flush=True)

    # ---- 2) EVAL（冻结、与训练+cal 不相交）：rl vs strong + greedy 基线 ----
    eval_streams = gen_streams("comp-eval", M_eval)
    rl_ev, rl_surv = _play_rl_streams(net, eng, eval_streams, transform)
    str_ev, str_surv = [], []
    grd_ev = []
    for s in eval_streams:
        st, ss = _play_baseline_stream(eng, s, "strong"); str_ev.append(st); str_surv.append(ss)
        grd_ev.append(_play_baseline_stream(eng, s, "greedy")[0])
    d_rs = [rl_ev[i] - str_ev[i] for i in range(M_eval)]    # rl − strong（主判据）
    d_rg = [rl_ev[i] - grd_ev[i] for i in range(M_eval)]    # rl − greedy（碾压检）
    st_rs = _paired_stats(d_rs); st_rg = _paired_stats(d_rg)

    # ---- 3) TOST verdict（90% CI vs ±Δ_eq；§6 路由）----
    lo90, hi90 = st_rs["ci90"]
    if lo90 > DELTA_EQ:
        verdict = "RL≫strong"          # §6(i) 天花板被低估 → EVPI 占比上修
    elif hi90 < -DELTA_EQ:
        verdict = "RL≪strong"          # §6(iii) 预算内未达天花板 → inconclusive（永不当结论）
    elif lo90 >= -DELTA_EQ and hi90 <= DELTA_EQ:
        verdict = "RL≈strong"          # §6(ii) TOST 等价 → 天花板两方法佐证（措辞降级：共享候选池）
    else:
        verdict = "inconclusive"        # CI 跨 Δ_eq 边界：既不等价也不显著优/劣

    res = {"transform": transform, "T": T, "GAMMA": GAMMA, "ckpt": ckpt, "beam_B": beam_B,
           "M_cal": M_cal, "M_eval": M_eval, "DELTA_EQ": DELTA_EQ,
           "strong_mean_cal": strong_mean_cal, "sigma_diff_cal": sd,
           "tost_power_ok": bool(power_ok),
           "rl_mean": sum(rl_ev) / M_eval, "strong_mean": sum(str_ev) / M_eval,
           "greedy_mean": sum(grd_ev) / M_eval,
           "rl_survival_mean": sum(rl_surv) / M_eval, "strong_survival_mean": sum(str_surv) / M_eval,
           "rl_minus_strong": st_rs, "rl_minus_greedy": st_rg,
           "verdict": verdict, "wall_s": time.time() - t0}

    print(f"\n[compete] === EVAL (M={M_eval}, CRN paired, T={T}) ===", flush=True)
    print(f"  rl_mean={res['rl_mean']:.1f}  strong_mean={res['strong_mean']:.1f}  "
          f"greedy_mean={res['greedy_mean']:.1f}", flush=True)
    print(f"  rl-strong: mean={st_rs['mean']:.1f}  block-SE={st_rs['block_se']:.1f}  "
          f"95%CI=[{st_rs['ci95'][0]:.1f},{st_rs['ci95'][1]:.1f}]  "
          f"90%CI=[{lo90:.1f},{hi90:.1f}]", flush=True)
    print(f"  rl-greedy: mean={st_rg['mean']:.1f}  95%CI=[{st_rg['ci95'][0]:.1f},{st_rg['ci95'][1]:.1f}]"
          f"  (碾压检：下界>0 => rl 显著强于盲贪心)", flush=True)
    print(f"  Δ_eq=±{DELTA_EQ:.1f}  → VERDICT = {verdict}", flush=True)
    print(f"  ({res['wall_s']:.0f}s)", flush=True)

    try:
        allres = json.load(open(out_json, encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        allres = {}
    allres[transform] = res
    json.dump(allres, open(out_json, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"-> wrote {out_json}", flush=True)
    return res


# ====================================================================
# 可用工具：最强策略（价值引导前瞻 + 训练 V）的单步推荐 + 整局演示
# ====================================================================
def _print_board8(board, title=""):
    if title:
        print(title, flush=True)
    print("   " + " ".join(str(c) for c in range(8)), flush=True)
    for r in range(8):
        row = "".join("█" if board & (1 << (r * 8 + c)) else "·" for c in range(8))
        print(f"{r}  " + " ".join(row), flush=True)


def _mask_rc(mask):
    cells = [(i // 8, i % 8) for i in range(64) if mask & (1 << i)]
    r0 = min(r for r, _ in cells); c0 = min(c for _, c in cells)
    return r0, c0, cells


def best_move(net, engine, board, combo, hand, k, transform, D=2, S=30, B=12):
    """最强策略单步推荐：给 (board,combo) + 3 块手牌 + 剩余轮 k，返回最佳放置。
    用价值引导前瞻（beam 候选 → 各 S 条 D-手 rollout + 学习 V 尾值 → 选最高）。
    返回 {afterstate,(board,combo), hand_score, placements:[(piece_id,row,col,cells)], value}。"""
    fast = engine.fast
    cands = fast.beam_hand_path(board, combo, hand, B)
    if not cands:
        return None
    NT = engine.NUM_TYPES
    leaf_b, leaf_c, leaf_kl, roll_sc, owner = [], [], [], [], []
    for ci, (nb, nc, hs, path) in enumerate(cands):
        if D == 0:
            leaf_b.append(nb); leaf_c.append(float(nc)); leaf_kl.append(max(k - 1, 0))
            roll_sc.append(0.0); owner.append(ci)
        else:
            for s in range(S):
                rng = random.Random(f"bestmove-{ci}-{s}")
                fut = [rng.randrange(NT) for _ in range(3 * D)]
                rscore, lb, lc, nd = _roll_leaf(fast, nb, nc, fut)
                leaf_b.append(lb); leaf_c.append(float(lc)); leaf_kl.append(max(k - 1 - nd, 0))
                roll_sc.append(rscore); owner.append(ci)
    with torch.no_grad():
        kt = torch.tensor([float(x) for x in leaf_kl], device=DEVICE)
        vleaf = decode_vtot(net(board_planes_fast8(leaf_b),
                                torch.tensor(leaf_c, device=DEVICE), kt / T_MAX), kt, transform).tolist()
    agg = [0.0] * len(cands); cnt = [0] * len(cands)
    for j, ci in enumerate(owner):
        agg[ci] += roll_sc[j] + vleaf[j]; cnt[ci] += 1
    best_ci, best_val = 0, -1e30
    for ci, (nb, nc, hs, path) in enumerate(cands):
        val = hs + agg[ci] / max(cnt[ci], 1)
        if val > best_val:
            best_val, best_ci = val, ci
    nb, nc, hs, path = cands[best_ci]
    placements = []
    for (hidx, mask) in path:
        pid = hand[hidx]
        r0, c0, cells = _mask_rc(mask)
        placements.append({"piece_id": pid, "row": r0, "col": c0, "cells": cells})
    return {"board": nb, "combo": nc, "hand_score": hs, "value": best_val, "placements": placements}


def play_game(ckpt, seed=0, T=50, D=2, S=30, B=12, transform="rate", hidden=128, show=True):
    """用最强策略（价值引导前瞻 + 训练 V）玩整局，逐回合显示。返回 (total, survived)。"""
    eng = Engine8()
    net = RateNet(hidden=hidden).to(DEVICE)
    net.load_state_dict(torch.load(ckpt, map_location=DEVICE))
    rng = random.Random(f"play-{seed}")
    board, combo, total = 0, 0, 0.0
    if show:
        print(f"=== 最强策略演示 | ckpt={ckpt} | D={D} S={S} B={B} | seed={seed} ===", flush=True)
    for rnd in range(T):
        k = T - rnd
        hand = eng.sample_hand(rng)
        mv = best_move(net, eng, board, combo, hand, k, transform, D, S, B)
        if mv is None:
            if show:
                print(f"\n[第 {rnd+1} 轮] 手牌 {hand} 无处可放 → 游戏结束", flush=True)
            return total, rnd
        if show:
            print(f"\n[第 {rnd+1}/{T} 轮] 手牌(piece_id)={hand}  combo={combo}", flush=True)
            for p in mv["placements"]:
                print(f"    放 piece {p['piece_id']} 于 (行{p['row']},列{p['col']})", flush=True)
            print(f"    本手得分 +{mv['hand_score']:.0f}", flush=True)
        board, combo, total = mv["board"], mv["combo"], total + mv["hand_score"]
        if show:
            _print_board8(board)
            print(f"    累计得分 = {total:.0f}", flush=True)
    if show:
        print(f"\n=== 完局！存活 {T}/{T} 轮，总分 = {total:.0f} ===", flush=True)
    return total, T


# ====================================================================
# 价值引导前瞻：前瞻搜索 + 学习 V 尾值（训练 + 搜索结合 = 最强可玩策略候选）
# benchmark 洞察：拿高分的杠杆 = 降低未来价值估计方差（S↑）；V 提供零方差尾值替代部分 rollout。
# ====================================================================
def _roll_leaf(fast, board, combo, future):
    """greedy(启发式) 玩 future（展平 piece 列表），返回 (rollout_score, leaf_board, leaf_combo, hands_done)。"""
    total = 0.0; nd = 0
    for i in range(0, len(future), 3):
        for pid in future[i:i + 3]:
            best = None
            for m in fast.PLACE[pid]:
                if board & m:
                    continue
                nb, cl, empty = fast.apply_mask(board, m)
                pts, nc = fast.score_placement(fast.SCORING, fast.NCELLS[pid], cl, empty, combo)
                val = pts + fast.heuristic(nb)
                if best is None or val > best[0]:
                    best = (val, nb, nc, pts)
            if best is None:
                return total, board, combo, nd          # rollout 中途死
            _, board, combo, pts = best
            total += pts
        nd += 1
    return total, board, combo, nd


def play_vlookahead_stream(net, engine, stream, transform, D, S, B, seed_idx, use_heur=False):
    """价值引导前瞻玩一条固定发牌流（CRN）。每候选末态：S 条 D-手 greedy rollout 累加真分，
    叶子加学习 V(leaf, 剩余轮) 当尾值 → value = hand_score + mean_S[roll + V_leaf]。
    D=0 → 纯 greedy-on-V。返回 (total, survived)。"""
    fast = engine.fast
    board, combo, total = 0, 0, 0.0
    T = len(stream)
    NT = engine.NUM_TYPES
    for move, hand in enumerate(stream):
        k = T - move
        cands = fast.beam_hand(board, combo, hand, B)
        if not cands:
            return total, move
        leaf_b, leaf_c, leaf_kl, roll_sc, owner = [], [], [], [], []
        for ci, (nb, nc, hs) in enumerate(cands):
            if D == 0:                                   # 纯 V：叶子=候选末态本身，剩 k-1 轮
                leaf_b.append(nb); leaf_c.append(float(nc)); leaf_kl.append(max(k - 1, 0))
                roll_sc.append(0.0); owner.append(ci)
            else:
                for s in range(S):
                    rng = random.Random(f"vla-{seed_idx}-{move}-{ci}-{s}")
                    fut = [rng.randrange(NT) for _ in range(3 * D)]
                    rsc, lb, lc, nd = _roll_leaf(fast, nb, nc, fut)
                    leaf_b.append(lb); leaf_c.append(float(lc)); leaf_kl.append(max(k - 1 - nd, 0))
                    roll_sc.append(rsc); owner.append(ci)
        with torch.no_grad():                            # 批量 V 评叶子（剩余轮各异→逐元素 ×k）
            kt = torch.tensor([float(x) for x in leaf_kl], device=DEVICE)
            kn = kt / T_MAX
            vleaf = decode_vtot(net(board_planes_fast8(leaf_b),
                                    torch.tensor(leaf_c, device=DEVICE), kn), kt, transform).tolist()
        # 聚合每候选的 value = hand_score(+heur) + mean_S[roll + V_leaf]
        agg = [0.0] * len(cands); cnt = [0] * len(cands)
        for j, ci in enumerate(owner):
            agg[ci] += roll_sc[j] + vleaf[j]; cnt[ci] += 1
        best_ci, best_val = 0, -1e30
        for ci, (nb, nc, hs) in enumerate(cands):
            val = hs + agg[ci] / max(cnt[ci], 1) + (fast.heuristic(nb) if use_heur else 0.0)
            if val > best_val:
                best_val, best_ci = val, ci
        nb, nc, hs = cands[best_ci]
        board, combo, total = nb, nc, total + hs
    return total, T


def vbench(transform, ckpt, hidden=128, T=50, M=200, configs=None, out_json="rl8_vbench.json"):
    """对比 strong / 纯 lookahead / 价值引导前瞻（用 ckpt 的 V）在同一 CRN 流上的实战得分。
    找"训练+搜索"的最强可玩策略。configs=[(label,D,S,B)]，D=0 为纯 greedy-on-V。"""
    eng = Engine8()
    net = RateNet(hidden=hidden).to(DEVICE)
    net.load_state_dict(torch.load(ckpt, map_location=DEVICE))
    if configs is None:
        configs = [("vla D0(greedy-on-V)", 0, 1, 12), ("vla D1 S10", 1, 10, 12),
                   ("vla D2 S10", 2, 10, 12), ("vla D3 S10", 3, 10, 12)]
    streams = []
    for m in range(M):
        rng = random.Random(f"bench-{m}")            # 与 bench_players 同 tag → 同流可跨文件比
        streams.append([[rng.randrange(eng.NUM_TYPES) for _ in range(3)] for _ in range(T)])
    print(f"=== vbench | ckpt={ckpt} | M={M} CRN | T={T} ===", flush=True)

    def block_stats(xs, nb=16):
        blk = len(xs) // nb
        bm = [sum(xs[i * blk:(i + 1) * blk]) / blk for i in range(nb)]
        mean = sum(bm) / nb
        se = (sum((x - mean) ** 2 for x in bm) / (nb - 1) / nb) ** 0.5
        return mean, se

    res = {}
    # strong 基线
    strong_tot = []
    for i in range(M):
        b, c, tot = 0, 0, 0.0
        for hand in streams[i]:
            sc, b, c, alive = eng.fast.strong_hand(b, c, hand, B=12)
            if not alive:
                break
            tot += sc
        strong_tot.append(tot)
    sm, sse = block_stats(strong_tot)
    res["strong B12"] = {"mean": sm, "se": sse}
    print(f"[strong B12          ] mean={sm:7.1f} ± {sse:5.1f}", flush=True)

    for (label, D, S, B) in configs:
        t0 = time.time()
        rows = [play_vlookahead_stream(net, eng, streams[i], transform, D, S, B, i) for i in range(M)]
        tots = [r[0] for r in rows]; survs = [r[1] for r in rows]
        mean, se = block_stats(tots); smean, _ = block_stats([float(x) for x in survs])
        diffs = [tots[i] - strong_tot[i] for i in range(M)]
        dmean, dse = block_stats(diffs)
        res[label] = {"mean": mean, "se": se, "survival": smean,
                      "vs_strong": dmean, "vs_strong_se": dse, "wall_s": time.time() - t0}
        flag = "✓更强" if dmean - 2 * dse > 0 else ("✗更弱" if dmean + 2 * dse < 0 else "≈")
        print(f"[{label:20s}] mean={mean:7.1f} ± {se:5.1f}  surv={smean:4.1f}  "
              f"Δstrong={dmean:+7.1f}±{dse:.1f} {flag}  ({res[label]['wall_s']:.0f}s)", flush=True)

    best = max(res, key=lambda k: res[k]["mean"])
    print(f"\n-> 最强 = {best} (mean={res[best]['mean']:.1f})", flush=True)
    json.dump(res, open(out_json, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"-> wrote {out_json}", flush=True)
    return res


# ====================================================================
# §5 正交性诊断：beam top-B 候选池是否系统性裁掉 rl-V 偏好的落点
# （RL≪strong 时验负结果非"同源候选池假象"：rl 选不到的落点 ≠ rl 真弱）
# ====================================================================
def _full_afterstates(engine, board, combo, hand, cap=120000):
    """无 beam 剪枝**全枚举**手内 3 块顺序×落点（combo 贯穿），返回去重 afterstate
    [(board',combo',hand_score_max)]。超 cap → 返回 None（该态太大，调用方记 skip，§no-silent-cap）。"""
    f = engine.fast
    states = [(board, combo, 0.0, ())]
    for _ in range(3):
        cand = []
        for (b, c, sc, used) in states:
            for i in range(3):
                if i in used:
                    continue
                pid = hand[i]
                for m in f.PLACE[pid]:
                    if b & m:
                        continue
                    nb, cl, empty = f.apply_mask(b, m)
                    pts, nc = f.score_placement(f.SCORING, f.NCELLS[pid], cl, empty, c)
                    cand.append((nb, nc, sc + pts, used + (i,)))
            if len(cand) > cap:
                return None
        if not cand:
            return []
        states = cand
    best = {}                                   # 去重 (board,combo)：同末态不同 order 取最高 hand_score
    for (b, c, sc, _) in states:
        if (b, c) not in best or sc > best[(b, c)]:
            best[(b, c)] = sc
    return [(b, c, sc) for (b, c), sc in best.items()]


def orthogonality_diag(transform, ckpt, hidden=128, T=50, n_rollouts=12,
                       cap=120000, out_json="rl8_ortho.json"):
    """§5：采 strong-rollout 中局态 × 随机手，比 rl-V-argmax over **full-enum** vs over **beam top-B**。
    报：full-argmax 落在 beam top-B 外的比例（=beam 裁掉 rl 偏好落点）+ rl 自身 V 看 full 比 beam 高多少。
    比例低 + V 差小 ⇒ 候选池公平 ⇒ RL≪strong 非同源池假象（负结果稳健）。"""
    eng = Engine8()
    net = RateNet(hidden=hidden).to(DEVICE)
    net.load_state_dict(torch.load(ckpt, map_location=DEVICE))
    print(f"=== §5 正交性诊断 | transform={transform} T={T} | ckpt={ckpt} ===", flush=True)

    def vscore(cands, kleft):
        """rl 对候选打分 hand_score + γ·V(afterstate,kleft-1)，返回 (best_idx, best_val, vals)。"""
        if kleft - 1 == 0:
            vt = [0.0] * len(cands)
        else:
            with torch.no_grad():
                kn = torch.full((len(cands),), (kleft - 1) / T_MAX, device=DEVICE)
                vt = decode_vtot(net(board_planes_fast8([b for b, _, _ in cands]),
                                     torch.tensor([float(c) for _, c, _ in cands], device=DEVICE), kn),
                                 kleft - 1, transform).tolist()
        vals = [cands[i][2] + GAMMA * vt[i] for i in range(len(cands))]
        bi = max(range(len(cands)), key=lambda i: vals[i])
        return bi, vals[bi]

    n_eval = n_cut = n_skip = n_diff = 0
    vgaps = []
    for r in range(n_rollouts):
        rng = random.Random(f"ortho-{r}")
        board, combo = 0, 0
        for i in range(T):
            kdec = T - i                                   # 此刻面对新手，剩 kdec 轮
            hand = eng.sample_hand(rng)
            beam = eng.round_candidates(board, combo, hand)   # top-B=12（rl 实际可选集）
            if not beam:
                break
            # 仅在中局诊断（i≥8，棋盘渐满→full 枚举可控）；早盘 full 太大跳过统计但继续推进
            if i >= 8 and kdec >= 2:
                full = _full_afterstates(eng, board, combo, hand, cap)
                if full is None:
                    n_skip += 1
                elif full:
                    n_eval += 1
                    bi_b, v_b = vscore(beam, kdec)
                    bi_f, v_f = vscore(full, kdec)
                    beam_keys = {(b, c) for b, c, _ in beam}
                    pick_full = (full[bi_f][0], full[bi_f][1])
                    if pick_full not in beam_keys:
                        n_cut += 1                          # rl 的 full-偏好落点不在 beam top-B 里
                    if pick_full != (beam[bi_b][0], beam[bi_b][1]):
                        n_diff += 1
                    vgaps.append(v_f - v_b)                 # rl 自身 V 看：full 最优比 beam 最优高多少
            # 用 strong 推进 rollout（采的是 strong 触及的中局分布）
            sc, board, combo, alive = eng.fast.strong_hand(board, combo, hand, B=eng.beam_B)
            if not alive:
                break

    cut_frac = n_cut / n_eval if n_eval else 0.0
    diff_frac = n_diff / n_eval if n_eval else 0.0
    mean_vgap = sum(vgaps) / len(vgaps) if vgaps else 0.0
    # vgap 相对量纲：用 beam 最优 V 的典型量级归一（粗略，仅 advisory）
    res = {"transform": transform, "T": T, "GAMMA": GAMMA, "n_eval": n_eval, "n_skip_cap": n_skip,
           "cut_frac": cut_frac, "argmax_diff_frac": diff_frac, "mean_vgap_full_minus_beam": mean_vgap,
           "verdict": "POOL_FAIR" if cut_frac < 0.10 else "POOL_BIASED"}
    print(f"\n[ortho] === §5 VERDICT (n_eval={n_eval}, skip_cap={n_skip}) ===", flush=True)
    print(f"  beam 裁掉 rl-full-偏好落点比例 cut_frac={cut_frac:.2%}  (阈<10% ⇒ 池公平)", flush=True)
    print(f"  full vs beam argmax 不同比例={diff_frac:.2%}  mean V_gap(full-beam)={mean_vgap:.2f}", flush=True)
    print(f"  → {res['verdict']}（POOL_FAIR ⇒ RL≪strong 非同源候选池假象，负结果稳健）", flush=True)
    try:
        allres = json.load(open(out_json, encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        allres = {}
    allres[transform] = res
    json.dump(allres, open(out_json, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"-> wrote {out_json}", flush=True)
    return res


# ====================================================================
# §3.0 迁移 gate：rl8 CNN+buffer+rate 变换 在 4×4 引擎复现 bdp_T，rel<5% 才放行
# ====================================================================
def migration_gate(**train_kw):
    eng = Engine4()
    print(f"=== §3.0 迁移 gate | 4×4 引擎 × 8×8-CNN | 靶 bdp_T(8)={BDP_T[8]} "
          f"(16)={BDP_T[16]} ===", flush=True)
    res = train_modeT(eng, **train_kw)
    out = {}
    pass_all = True
    for T in (8, 16):
        v = res["v8"] if T == 8 else res["v16"]
        rel = abs(v - BDP_T[T]) / BDP_T[T]
        ok = rel < 0.05
        pass_all &= ok
        print(f"[迁移 gate] T={T}  Vtot(empty,{T})={v:.4f}  靶={BDP_T[T]}  "
              f"rel={rel:.4%}  阈<5% → {'PASS' if ok else 'FAIL'}", flush=True)
        out[f"T{T}"] = {"v_total": v, "target": BDP_T[T], "rel": rel, "pass": bool(ok)}
    out["pass"] = bool(pass_all)
    out["train"] = {k: res[k] for k in ("v8", "v16", "buf_size", "sweeps",
                                        "wall_s", "plateau", "wall_budget_exceeded")}
    return out


def _smoke():
    """秒级冒烟：网络 forward + 编码 + 一手候选打通（不训练）。"""
    net = RateNet().to(DEVICE)
    npar = sum(p.numel() for p in net.parameters())
    eng = Engine4()
    pe = board_planes_batch([0, 1, 7], 4)
    r = net(pe, torch.zeros(3, device=DEVICE), torch.full((3,), 0.5, device=DEVICE))
    assert r.shape == (3,), r.shape
    cands = eng.round_candidates(0, 0, [0])
    print(f"[smoke] RateNet params={npar}  forward OK shape={tuple(r.shape)}  "
          f"4×4 round_candidates(empty,piece0)={len(cands)} 落点", flush=True)
    eng8 = Engine8()
    c8 = eng8.round_candidates(0, 0, [0, 1, 2])
    print(f"[smoke] 8×8 beam_hand(empty,[0,1,2])={len(c8)} afterstate 候选", flush=True)


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "smoke"
    print(f"=== rl8.py | torch {torch.__version__} | mode={mode} ===", flush=True)
    if mode == "smoke":
        _smoke()
    elif mode == "gate":
        # 迁移 gate 默认轻量参数；过不了再调（loop 迭代）
        res = migration_gate()
        try:
            allres = json.load(open("rl8_gate.json"))
        except (FileNotFoundError, json.JSONDecodeError):
            allres = {}
        allres["migration_gate"] = res
        json.dump(allres, open("rl8_gate.json", "w"), ensure_ascii=False, indent=2)
        print("\n-> wrote rl8_gate.json", flush=True)
        print(json.dumps(res, ensure_ascii=False, indent=2), flush=True)
    elif mode in ("train8", "check8"):
        # 8×8 生产 horizon：把 T_MAX 设为 HORIZON_T（默认 50），网络 k 归一化用之。
        import os
        T_MAX = int(sys.argv[3]) if len(sys.argv) > 3 else 50
        # 补20：8×8 γ-折扣训练稳定器（无折扣 T=50 发散）。判强/天花板仍用无折扣实战得分。
        # γ 可经 RL8_GAMMA 环境变量覆盖（默认 0.95）；补21+competence 后探更高 γ 治近视早死。
        GAMMA = float(os.environ.get("RL8_GAMMA", "0.95"))
        transform = sys.argv[2] if len(sys.argv) > 2 else "logv"
        print(f"[8×8] GAMMA={GAMMA}（训练稳定器；rollout 实战得分仍无折扣累加）", flush=True)
        # γ 编进 ckpt 名防覆盖（0.95 保原名向后兼容）。
        gtag = "" if abs(GAMMA - 0.95) < 1e-9 else f"_g{int(round(GAMMA*100))}"
        ckpt = f"/tmp/rl8_8x8_{transform}_T{T_MAX}{gtag}.pt"
        if mode == "train8":
            # §3.6★ 候选变换在 8×8 训到 plateau（off-policy 补采覆盖尾部）。可断点续训。
            print(f"=== 8×8 mode-T 训练 | transform={transform} T_MAX={T_MAX} | ckpt={ckpt} ===", flush=True)
            smoke = len(sys.argv) > 4 and sys.argv[4] == "smoke"
            # 补17：lr 2e-3→5e-4 + grad_clip=1.0（+ decode_vtot logv clamp）镇 value blow-up→NaN。
            kw = dict(transform=transform, hidden=128, lr=5e-4, inner_steps=3,
                      ckpt_path=ckpt, ckpt_every=10, offpolicy=(20, 20),
                      n_rollouts=200, eps=0.3, hand_samples=8, grad_clip=1.0)
            if smoke:
                kw.update(max_sweeps=4, min_sweeps=2, wall_budget_s=600,
                          offpolicy=(2, 2), n_rollouts=20)
            else:
                # 3h 续训 chunk（补2-4 模式：resume_ckpt 累积跨 loop 轮次）。
                # min_sweeps≥T_MAX（补gate：mode-T 是后向归纳，早停=undershoot）。
                # recollect 关闭（补16）：off-policy(strong+seer)+初始 self-play 已覆盖，
                # recollect 会让 buffer→eval-dst 无界长大 → MPS OOM。固定 buffer 更稳。
                kw.update(max_sweeps=1500, min_sweeps=max(120, T_MAX),
                          wall_budget_s=10800, resume_ckpt=ckpt, recollect_every=10**9)
            res = train_modeT(Engine8(), **kw)
            print(f"-> 8×8 train DONE v8={res['v8']:.2f} v16={res['v16']:.2f} "
                  f"sweeps={res['sweeps']} plateau={res['plateau']} ckpt={ckpt}", flush=True)
        else:  # check8
            M = int(sys.argv[4]) if len(sys.argv) > 4 else 2000
            nprobes = int(sys.argv[5]) if len(sys.argv) > 5 else 32
            selfcons_highk(transform, ckpt, hidden=128, M=M, n_probes=nprobes)
    elif mode == "compete":
        # §5/§6 Phase 3：rl-greedy-on-V vs strong（CRN 配对）= 最强策略 verdict。
        # 用法：python rl8.py compete [transform=rate] [T=50] [M_eval=2000]
        import os
        transform = sys.argv[2] if len(sys.argv) > 2 else "rate"
        T_MAX = int(sys.argv[3]) if len(sys.argv) > 3 else 50
        GAMMA = float(os.environ.get("RL8_GAMMA", "0.95"))   # 与 train8 一致；RL8_GAMMA 覆盖
        M_eval = int(sys.argv[4]) if len(sys.argv) > 4 else 2000
        beam_B = int(sys.argv[5]) if len(sys.argv) > 5 else 12
        gtag = "" if abs(GAMMA - 0.95) < 1e-9 else f"_g{int(round(GAMMA*100))}"
        ckpt = f"/tmp/rl8_8x8_{transform}_T{T_MAX}{gtag}.pt"
        # B≠12 写独立 JSON 防覆盖基线（§5 disambiguation）
        oj = "rl8_competence.json" if beam_B == 12 else f"rl8_competence_B{beam_B}.json"
        print(f"[8×8] GAMMA={GAMMA} beam_B={beam_B}（选点折扣；判强用无折扣实战得分）", flush=True)
        res = competence_gate(transform, ckpt, hidden=128, T=T_MAX, M_eval=M_eval,
                              beam_B=beam_B, out_json=oj)
        print(f"\n-> competence VERDICT = {res['verdict']}", flush=True)
    elif mode == "trainpi":
        # 近似策略迭代修早死：python rl8.py trainpi [transform=rate] [T=50] [n_outer=6]
        # 默认从 g98 网络热启、γ=0.98；训完自动 competence 评估对比 strong。
        import os
        transform = sys.argv[2] if len(sys.argv) > 2 else "rate"
        T_MAX = int(sys.argv[3]) if len(sys.argv) > 3 else 50
        n_outer = int(sys.argv[4]) if len(sys.argv) > 4 else 6
        GAMMA = float(os.environ.get("RL8_GAMMA", "0.98"))
        gtag = "" if abs(GAMMA - 0.95) < 1e-9 else f"_g{int(round(GAMMA*100))}"
        seed_ckpt = f"/tmp/rl8_8x8_{transform}_T{T_MAX}{gtag}.pt"
        if not os.path.exists(seed_ckpt):
            seed_ckpt = None
        print(f"[8×8] GAMMA={GAMMA} trainpi n_outer={n_outer} seed={seed_ckpt}", flush=True)
        out = train_onpolicy(transform, T=T_MAX, n_outer=n_outer, seed_ckpt=seed_ckpt)
        res = competence_gate(transform, out["ckpt"], hidden=128, T=T_MAX,
                              M_eval=2000, out_json="rl8_competence_pi.json")
        print(f"\n-> PI competence VERDICT = {res['verdict']}  rl_mean={res['rl_mean']:.1f} "
              f"(vs strong {res['strong_mean']:.1f})", flush=True)
    elif mode == "play":
        # 最强策略演示：python rl8.py play [seed=0] [S=30] [ckpt=models/strongest_v.pt]
        import os
        T_MAX = 50
        GAMMA = float(os.environ.get("RL8_GAMMA", "0.98"))
        seed = int(sys.argv[2]) if len(sys.argv) > 2 else 0
        S = int(sys.argv[3]) if len(sys.argv) > 3 else 30
        ckpt = sys.argv[4] if len(sys.argv) > 4 else "models/strongest_v.pt"
        play_game(ckpt, seed=seed, T=50, D=2, S=S, B=12, transform="rate", show=True)
    elif mode == "vbench":
        # 价值引导前瞻对比：python rl8.py vbench [transform=rate] [T=50] [M=200] [ckpt_tag=pi|g98|g95]
        import os
        transform = sys.argv[2] if len(sys.argv) > 2 else "rate"
        T_MAX = int(sys.argv[3]) if len(sys.argv) > 3 else 50
        M = int(sys.argv[4]) if len(sys.argv) > 4 else 200
        tag = sys.argv[5] if len(sys.argv) > 5 else "pi"
        GAMMA = float(os.environ.get("RL8_GAMMA", "0.98"))
        ckpt = {"pi": f"/tmp/rl8_8x8_{transform}_T{T_MAX}_pi.pt",
                "g98": f"/tmp/rl8_8x8_{transform}_T{T_MAX}_g98.pt",
                "g95": f"/tmp/rl8_8x8_{transform}_T{T_MAX}.pt"}.get(tag, tag)
        print(f"[8×8] GAMMA={GAMMA} vbench ckpt={ckpt}", flush=True)
        vbench(transform, ckpt, hidden=128, T=T_MAX, M=M)
    elif mode == "ortho":
        # §5 正交性诊断：用法 python rl8.py ortho [transform=rate] [T=50] [n_rollouts=12]
        import os
        transform = sys.argv[2] if len(sys.argv) > 2 else "rate"
        T_MAX = int(sys.argv[3]) if len(sys.argv) > 3 else 50
        GAMMA = float(os.environ.get("RL8_GAMMA", "0.95"))
        nroll = int(sys.argv[4]) if len(sys.argv) > 4 else 12
        gtag = "" if abs(GAMMA - 0.95) < 1e-9 else f"_g{int(round(GAMMA*100))}"
        ckpt = f"/tmp/rl8_8x8_{transform}_T{T_MAX}{gtag}.pt"
        print(f"[8×8] GAMMA={GAMMA}", flush=True)
        orthogonality_diag(transform, ckpt, hidden=128, T=T_MAX, n_rollouts=nroll)
