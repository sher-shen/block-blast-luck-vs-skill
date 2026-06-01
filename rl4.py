"""rl4.py — 4×4 afterstate-FVI 双 gate 认证 harness（torch，仅此文件依赖 torch）。

定位（PHASE_EXEC_PLAN.md ①）：在有精确 DP 真值的 4×4 上先证明 afterstate 拟合值迭代
(FVI) pipeline 可信，再烧算力上 8×8。没有这一步，8×8 打不过 beam-strong 时分不清
"到天花板了"还是"网没训好"（弱 agent 陷阱）。

两套折扣制度（务必分开——审核2/3 核心 apples-to-oranges）：
  - mode-γ：γ=0.95 折扣无限期，复现 dp4 收敛 V*(empty)=3.9157。in=16。
  - mode-T：无折扣有限-horizon-T（V 条件于 rounds-left k），复现 backward_dp_T。in=17。

不认证项（README/log 须声明，类 R3 的 F3）：4×4 是单块/回合、线性计分（无 combo）。
双 gate 认证 {值网络 + FVI 循环 + γ/T 两种 backup + 真值复现 + 位移检查}，
**不**认证 8×8 的 beam_hand 三块手牌候选枚举 × afterstate 集成（那层靠 8×8 competence gate）。

真值（dp4.py 现码现跑）：γ V*(empty,tol1e-5)=3.9157；bdp_T(8)=3.5934, bdp_T(16)=4.8637；
近视贪心 greedy_optimality_ratio=0.859；corr(heuristic4, V*)=0.870。

硬约定：print 全 flush=True；目标里不含 heuristic（warm-start 仿射映射只进 init）；
数值结论带 CI、避免 ratio-of-means（γ-策略 gate 用 paired-CRN bootstrap）。
"""

import json
import random
import sys
import time

import torch
import torch.nn as nn

import dp4
from dp4 import (K, PLACE4, apply4, heuristic4, legal_moves, reachable_boards,
                 value_iteration, backward_dp_T, horizon_for)
from oracle_analysis import bootstrap_ci

torch.manual_seed(0)
DEVICE = "cpu"  # MPS 对 16→64→64→1 无收益（PHASE_EXEC_PLAN §1）
NCELL = dp4.NCELL  # 16
T_MAX = 16         # mode-T 训到 k=16，gate 在 k∈{8,16}
GAMMA = 0.95

# 预注册真值（gate 靶）
V_STAR_EMPTY = 3.9157
BDP_T = {8: 3.5934, 16: 4.8637}


# ---------- target-transform 钩子（8×8 才换 log1p/Huber，见 ②a-F1） ----------
def transform(y):
    """4×4 reward 有界(≤8) → identity。8×8 combo 无界才换 log1p/Huber。"""
    return y


def _test_identity_transform():
    """单测：identity 时数值不变（钩子接口位的回归测试）。"""
    t = torch.tensor([0.0, 1.5, 8.0, -2.0])
    assert torch.allclose(transform(t), t), "transform(identity) 改了数值！"


# ---------- 网络：MLP in→64→64→1, ReLU, 无 BN ----------
class VNet(nn.Module):
    def __init__(self, in_dim, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


# ---------- board int → 16 维 {0,1} ----------
def board_bits(b):
    return [(b >> i) & 1 for i in range(NCELL)]


# ======================================================================
# 共享：reachable 集 + 转移表（src→dst, cl），向量化 FVI backup 用
# ======================================================================
def build_transitions():
    """返回 (boards, idx_of, X, src_t, dst_t, cl_t, group_t, N)。
    group_t = src_index*K + p，用于 scatter amax（每个 (b,p) 段取最优落点）。
    放不下的 (b,p) 段在 scatter 中保持 init=0（= value_iteration 的 best=0.0 约定）。"""
    R = sorted(reachable_boards())
    N = len(R)
    idx_of = {b: i for i, b in enumerate(R)}
    X = torch.tensor([board_bits(b) for b in R], dtype=torch.float32, device=DEVICE)
    src, dst, cl, grp = [], [], [], []
    for i, b in enumerate(R):
        for p in range(K):
            for (nb, c) in legal_moves(b, p):
                src.append(i); dst.append(idx_of[nb]); cl.append(float(c))
                grp.append(i * K + p)
    src_t = torch.tensor(src, dtype=torch.long, device=DEVICE)
    dst_t = torch.tensor(dst, dtype=torch.long, device=DEVICE)
    cl_t = torch.tensor(cl, dtype=torch.float32, device=DEVICE)
    grp_t = torch.tensor(grp, dtype=torch.long, device=DEVICE)
    return R, idx_of, X, src_t, dst_t, cl_t, grp_t, N


def segmax_backup(values_at_dst, cl_t, grp_t, N):
    """给定 V[dst]（每条转移的 continuation 值），算每个 board 的 backup：
    y(b) = (1/K) Σ_p max(0, max_落点(cl + cont))。
    scatter_reduce amax，段 init=0 → 自动实现 max(0,·) 与"放不下→0"。"""
    trans = cl_t + values_at_dst                                  # M_trans
    seg = torch.zeros(N * K, device=DEVICE)
    seg = seg.scatter_reduce(0, grp_t, trans, reduce="amax", include_self=True)
    return seg.view(N, K).sum(dim=1) / K                          # (N,)


# ======================================================================
# warm-start：heuristic4 仿射映射到 V* 量纲（R2-B3）
# ======================================================================
def affine_warmstart_target(R):
    """heuristic4 ∈ [−24,−8]（负，只供 ordering），V* ∈ [0.07,4.56]（正）→ 不能直拟。
    线性把 heuristic4 的 [min,max] 缩放到 [0, 4.6]（a>0，单调）。返回 (target_tensor, a, b0)。"""
    h = [heuristic4(b) for b in R]
    hmin, hmax = min(h), max(h)
    a = 4.6 / (hmax - hmin)            # a>0
    b0 = -a * hmin
    tgt = torch.tensor([a * x + b0 for x in h], dtype=torch.float32, device=DEVICE)
    return tgt, a, b0


# ======================================================================
# mode-γ：γ-折扣无限期 FVI
# ======================================================================
def train_gamma(max_sweeps=600, inner_steps=20, lr=1e-3, plateau_tol=0.001,
                plateau_K=8, warm_epochs=2, warm_steps=300, wall_budget_s=600,
                hidden=256, min_sweeps=60, lr_decay_at=300, lr_decay=0.3):
    t0 = time.time()
    R, idx_of, X, src_t, dst_t, cl_t, grp_t, N = build_transitions()
    print(f"[mode-γ] reachable N={N}  K={K}  transitions={src_t.numel()}", flush=True)
    probe_idx = [idx_of[0]] + list(range(0, N, max(1, N // 30)))[:30]
    probe_idx = sorted(set(probe_idx))
    empty_i = idx_of[0]

    net = VNet(16, hidden).to(DEVICE)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    mse = nn.MSELoss()

    # --- warm-start（仿射映射 heuristic4）---
    warm_tgt, a_aff, b_aff = affine_warmstart_target(R)
    init_affine = warm_tgt.detach().clone()
    warm_end_mse = None
    for ep in range(warm_epochs):
        for _ in range(warm_steps // warm_epochs):
            opt.zero_grad()
            loss = mse(net(X), transform(warm_tgt))
            loss.backward(); opt.step()
        warm_end_mse = loss.item()
    print(f"[mode-γ] warm-start done, end MSE={warm_end_mse:.4f} "
          f"(affine a={a_aff:.4f} b={b_aff:.4f})", flush=True)

    # --- FVI sweep-1 MSE 检查 → fallback 冷启（R2-B3 预注册逃生门）---
    with torch.no_grad():
        V = net(X)
        y1 = segmax_backup(GAMMA * V[dst_t], cl_t, grp_t, N)
        sweep1_mse = mse(net(X), transform(y1)).item()
    warm_used = True
    if warm_end_mse and sweep1_mse > 10 * warm_end_mse:
        print(f"[mode-γ] FALLBACK: sweep-1 MSE {sweep1_mse:.4f} > 10×warm "
              f"{warm_end_mse:.4f} → 冷启重训（保留逃生但记录）", flush=True)
        net = VNet(16, hidden).to(DEVICE)
        opt = torch.optim.Adam(net.parameters(), lr=lr)
        warm_used = False

    # --- 半梯度 FVI 循环 ---
    probe_hist = []
    converged = False; sweeps_done = 0
    for sweep in range(max_sweeps):
        with torch.no_grad():
            V = net(X)
            y = segmax_backup(GAMMA * V[dst_t], cl_t, grp_t, N)
        y = transform(y)
        for _ in range(inner_steps):
            opt.zero_grad()
            loss = mse(net(X), y)
            loss.backward(); opt.step()
        with torch.no_grad():
            pv = net(X)[probe_idx]
        probe_hist.append(pv)
        sweeps_done = sweep + 1
        if sweep == lr_decay_at:
            for pg in opt.param_groups:
                pg["lr"] *= lr_decay
            print(f"[mode-γ] lr decay @ sweep {sweep} → {opt.param_groups[0]['lr']:.1e}", flush=True)
        if sweep % 20 == 0 or sweep < 3:
            print(f"[mode-γ] sweep {sweep:3d}  V(empty)={pv[probe_idx.index(empty_i)]:.4f}"
                  f"  MSE={loss.item():.5f}", flush=True)
        # plateau：探针集末 K 段相对变化 < plateau_tol（须 ≥ min_sweeps，防早停）
        if sweep + 1 >= min_sweeps and len(probe_hist) > plateau_K:
            recent = torch.stack(probe_hist[-(plateau_K + 1):])
            rel = (recent[1:] - recent[:-1]).abs() / (recent[:-1].abs() + 1e-6)
            if rel.max().item() < plateau_tol:
                converged = True
                print(f"[mode-γ] PLATEAU @ sweep {sweep} (max rel {rel.max():.4%})", flush=True)
                break
        if time.time() - t0 > wall_budget_s:
            print(f"[mode-γ] WALL-CLOCK 超预算 {wall_budget_s}s → 停", flush=True)
            break

    with torch.no_grad():
        Vfinal = net(X)
    wall = time.time() - t0
    v_empty = float(Vfinal[empty_i])
    print(f"[mode-γ] DONE  V_net(empty)={v_empty:.4f}  sweeps={sweeps_done}  "
          f"wall={wall:.1f}s  warm_used={warm_used}  plateau={converged}", flush=True)

    Vcache = {b: float(Vfinal[i]) for i, b in enumerate(R)}
    return {"net": net, "R": R, "idx_of": idx_of, "X": X, "Vcache": Vcache,
            "Vfinal": Vfinal, "init_affine": init_affine, "probe_idx": probe_idx,
            "v_empty": v_empty, "sweeps": sweeps_done, "wall_s": wall,
            "warm_used": warm_used, "plateau": converged,
            "wall_budget_exceeded": wall > wall_budget_s}


# ======================================================================
# mode-T：无折扣有限-horizon-T，网络版后向归纳 FVI
# ======================================================================
def train_T(max_sweeps=600, inner_steps=20, lr=1e-3, plateau_tol=0.001,
            plateau_K=8, warm_epochs=2, warm_steps=300, wall_budget_s=1800,
            hidden=128, min_sweeps=80, lr_decay_at=200, lr_decay=0.3):
    t0 = time.time()
    R, idx_of, X16, src_t, dst_t, cl_t, grp_t, N = build_transitions()
    print(f"[mode-T] reachable N={N}  K={K}  transitions={src_t.numel()}  "
          f"T_MAX={T_MAX}", flush=True)
    empty_i = idx_of[0]

    # 17 维输入：每层 k 一份 (board16, k/T_MAX)。k=0..T_MAX。
    def inp(k):
        kc = torch.full((N, 1), k / T_MAX, device=DEVICE)
        return torch.cat([X16, kc], dim=1)
    inputs = {k: inp(k) for k in range(T_MAX + 1)}
    # 训练样本：把 k=0..T_MAX 全部 board 堆叠（k=0 锚 0）
    X_all = torch.cat([inputs[k] for k in range(T_MAX + 1)], dim=0)  # ((T+1)*N, 17)

    net = VNet(17, hidden).to(DEVICE)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    mse = nn.MSELoss()

    # warm-start：仿射映射，对所有 k 广播同一目标（k-agnostic init；FVI 后向让各 k 分化）
    warm_tgt, a_aff, b_aff = affine_warmstart_target(R)
    Y_warm = torch.cat([torch.zeros(N, device=DEVICE)] +     # k=0 锚 0
                       [warm_tgt for _ in range(T_MAX)], dim=0)
    warm_end_mse = None
    for ep in range(warm_epochs):
        for _ in range(warm_steps // warm_epochs):
            opt.zero_grad()
            loss = mse(net(X_all), transform(Y_warm))
            loss.backward(); opt.step()
        warm_end_mse = loss.item()
    print(f"[mode-T] warm-start done, end MSE={warm_end_mse:.4f}", flush=True)
    init_affine = warm_tgt.detach().clone()

    def build_targets():
        """对每层 k=1..T_MAX：y_k(b)=(1/K)Σ_p max(0,max_落点(cl + V_net(b',k-1)))。
        continuation 用 k-1 层；k-1==0 时 V≡0（硬锚，不查网）。返回 ((T+1)*N,) 目标。"""
        with torch.no_grad():
            VK = {}
            for j in range(T_MAX):                     # continuation 层 0..T_MAX-1
                VK[j] = torch.zeros(N, device=DEVICE) if j == 0 else net(inputs[j])
            ys = [torch.zeros(N, device=DEVICE)]        # k=0 锚 0
            for k in range(1, T_MAX + 1):
                cont = VK[k - 1]                         # rounds-left k-1 的 continuation
                yk = segmax_backup(cont[dst_t], cl_t, grp_t, N)
                ys.append(yk)
        return torch.cat(ys, dim=0)

    # sweep-1 fallback 检查
    with torch.no_grad():
        sweep1_mse = mse(net(X_all), transform(build_targets())).item()
    warm_used = True
    if warm_end_mse and sweep1_mse > 10 * warm_end_mse:
        print(f"[mode-T] FALLBACK: sweep-1 MSE {sweep1_mse:.4f} > 10×warm "
              f"{warm_end_mse:.4f} → 冷启", flush=True)
        net = VNet(17, hidden).to(DEVICE)
        opt = torch.optim.Adam(net.parameters(), lr=lr)
        warm_used = False

    probe_hist = []
    converged = False; sweeps_done = 0
    for sweep in range(max_sweeps):
        Y = transform(build_targets())
        for _ in range(inner_steps):
            opt.zero_grad()
            loss = mse(net(X_all), Y)
            loss.backward(); opt.step()
        with torch.no_grad():
            v8 = float(net(inputs[8])[empty_i])
            v16 = float(net(inputs[16])[empty_i])
        probe_hist.append(torch.tensor([v8, v16]))
        sweeps_done = sweep + 1
        if sweep == lr_decay_at:
            for pg in opt.param_groups:
                pg["lr"] *= lr_decay
            print(f"[mode-T] lr decay @ sweep {sweep} → {opt.param_groups[0]['lr']:.1e}", flush=True)
        if sweep % 20 == 0 or sweep < 3:
            print(f"[mode-T] sweep {sweep:3d}  V(empty,8)={v8:.4f}  "
                  f"V(empty,16)={v16:.4f}  MSE={loss.item():.5f}", flush=True)
        if sweep + 1 >= min_sweeps and len(probe_hist) > plateau_K:
            recent = torch.stack(probe_hist[-(plateau_K + 1):])
            rel = (recent[1:] - recent[:-1]).abs() / (recent[:-1].abs() + 1e-6)
            if rel.max().item() < plateau_tol:
                converged = True
                print(f"[mode-T] PLATEAU @ sweep {sweep} (max rel {rel.max():.4%})", flush=True)
                break
        if time.time() - t0 > wall_budget_s:
            print(f"[mode-T] WALL-CLOCK 超预算 {wall_budget_s}s → 停", flush=True)
            break

    # 建 VcacheT[k][b] 供在线 rollout
    with torch.no_grad():
        VcacheT = {0: {b: 0.0 for b in R}}
        for k in range(1, T_MAX + 1):
            vk = net(inputs[k])
            VcacheT[k] = {b: float(vk[i]) for i, b in enumerate(R)}
        v8 = float(net(inputs[8])[empty_i])
        v16 = float(net(inputs[16])[empty_i])
    wall = time.time() - t0
    print(f"[mode-T] DONE  V(empty,8)={v8:.4f}  V(empty,16)={v16:.4f}  "
          f"sweeps={sweeps_done}  wall={wall:.1f}s  warm_used={warm_used}  "
          f"plateau={converged}", flush=True)
    return {"net": net, "R": R, "VcacheT": VcacheT, "v8": v8, "v16": v16,
            "sweeps": sweeps_done, "wall_s": wall, "warm_used": warm_used,
            "plateau": converged, "wall_budget_exceeded": wall > wall_budget_s}


# ======================================================================
# γ-gate：值 + 策略（paired-CRN bootstrap，非点 ratio-of-means）
# ======================================================================
def gate_gamma(g, M=20000):
    Vcache = g["Vcache"]
    v_empty = g["v_empty"]
    L = horizon_for(GAMMA)
    # 真值 V*（用于 paired 比较的分母 online-opt）
    Vstar, _, _ = value_iteration(gamma=GAMMA, tol=1e-5)
    greedy_ret, opt_ret = [], []
    for s in range(M):
        rng = random.Random(f"dp4-{s}")
        seq = [rng.randrange(K) for _ in range(L)]
        greedy_ret.append(dp4.play_online(Vcache, seq, GAMMA, L))   # 对 V_net 贪心
        opt_ret.append(dp4.play_online(Vstar, seq, GAMMA, L))       # 对 V* 贪心(=π*)
    # paired-CRN bootstrap：ratio = mean(greedy)/mean(opt)，同 index 重采样
    n = len(greedy_ret)
    rng = random.Random("rl4-gamma-policy")
    reps = []
    for _ in range(2000):
        idx = [rng.randrange(n) for _ in range(n)]
        gm = sum(greedy_ret[i] for i in idx) / n
        om = sum(opt_ret[i] for i in idx) / n
        reps.append(gm / om if om else 0.0)
    reps.sort()
    ci_lo = reps[int(0.025 * 2000)]
    ci_hi = reps[int(0.975 * 2000) - 1]
    ratio_point = (sum(greedy_ret) / n) / (sum(opt_ret) / n)
    ratio_vs_vstar = (sum(greedy_ret) / n) / V_STAR_EMPTY      # 也报 /V*(3.9157)

    value_pass = abs(v_empty - V_STAR_EMPTY) < 0.05
    policy_pass = ci_lo >= 0.92
    print(f"[γ-gate 值]  V_net(empty)={v_empty:.4f}  |Δ|={abs(v_empty-V_STAR_EMPTY):.4f}"
          f"  阈<0.05  → {'PASS' if value_pass else 'FAIL'}", flush=True)
    print(f"[γ-gate 策略] greedy/opt ratio={ratio_point:.4f} "
          f"CI[{ci_lo:.4f},{ci_hi:.4f}]  CI下界阈≥0.92  → "
          f"{'PASS' if policy_pass else 'FAIL'}  (ratio/V*={ratio_vs_vstar:.4f})", flush=True)
    return {"value_pass": bool(value_pass), "policy_pass": bool(policy_pass),
            "v_net_empty": v_empty, "v_star_empty": V_STAR_EMPTY,
            "abs_delta": abs(v_empty - V_STAR_EMPTY),
            "policy_ratio_point": ratio_point, "policy_ci": [ci_lo, ci_hi],
            "policy_ratio_vs_vstar": ratio_vs_vstar, "M": M, "L": L}


# ======================================================================
# T-gate：值（T=8,16）+ 在线无折扣 T-rollout（M=50k + 精度护栏）
# ======================================================================
def play_T_rollout(VcacheT, seq, T):
    """无折扣 T-round rollout：b=0 起，对 V_net 贪心走 T 步，累加 cl，无 γ、无截断 L。
    放不下→forfeit 剩余 horizon（该步起 continuation=0），与 bdp_T 同约定。"""
    b = 0; tot = 0.0
    for i in range(T):
        k = T - i            # 含本步的 rounds-left
        p = seq[i]
        best = None
        for m in PLACE4[p]:
            if b & m:
                continue
            nb, cl = apply4(b, m)
            cont = VcacheT[k - 1].get(nb, 0.0)   # k-1 rounds-left continuation
            v = cl + cont
            if best is None or v > best[0]:
                best = (v, nb, cl)
        if best is None:
            break            # forfeit，剩余 0
        tot += best[2]; b = best[1]
    return tot


def gate_T(t, M=50000, nblocks=16):
    VcacheT = t["VcacheT"]
    per = M // nblocks
    out = {}
    value_pass_all = True; online_pass_all = True
    for T in (8, 16):
        target = BDP_T[T]
        v_net = t["v8"] if T == 8 else t["v16"]
        value_pass = abs(v_net - target) / target < 0.05
        value_pass_all &= value_pass
        # 在线 rollout，16 块 block-SE
        blk = []
        for bi in range(nblocks):
            acc = 0.0
            for s in range(bi * per, (bi + 1) * per):
                rng = random.Random(f"rlT-{s}")
                seq = [rng.randrange(K) for _ in range(T)]
                acc += play_T_rollout(VcacheT, seq, T)
            blk.append(acc / per)
        gm = sum(blk) / nblocks
        bse = (sum((x - gm) ** 2 for x in blk) / nblocks / nblocks) ** 0.5
        rel = abs(gm - target) / target
        online_pass = rel < 0.05
        guard_pass = 2 * bse < 0.025 * target           # 精度护栏 2·SE < 2.5%·bdp_T
        online_pass_all &= (online_pass and guard_pass)
        print(f"[T-gate 值] T={T}  V_net(empty,{T})={v_net:.4f}  靶={target}  "
              f"rel={abs(v_net-target)/target:.4%}  阈<5%  → "
              f"{'PASS' if value_pass else 'FAIL'}", flush=True)
        print(f"[T-gate 在线] T={T}  rollout mean={gm:.4f}±{bse:.4f}(SE) M={M}  "
              f"rel={rel:.4%}<5%→{'PASS' if online_pass else 'FAIL'}  "
              f"护栏 2·SE={2*bse:.4f}<{0.025*target:.4f}→"
              f"{'PASS' if guard_pass else 'FAIL'}", flush=True)
        out[f"T{T}"] = {"v_net": v_net, "target": target,
                        "value_rel": abs(v_net - target) / target,
                        "value_pass": bool(value_pass),
                        "online_mean": gm, "online_block_se": bse,
                        "online_rel": rel, "online_pass": bool(online_pass),
                        "guard_pass": bool(guard_pass)}
    out["value_pass"] = bool(value_pass_all)
    out["online_pass"] = bool(online_pass_all)
    return out


# ======================================================================
# 位移检查（条件 gate：仅在两 gate 皆 PASS 后评估）
# ======================================================================
def displacement_check(g):
    """PASS 必需(binding)=探针集 V 变化(V-单位) > τ_disp；
    辅助(advisory)=corr(V_net, heuristic4) < 0.92，≥0.92→USER_GATE 人工核查。
    τ_disp = 0.3 × RMS(V* − init_affine) over 探针集（V-单位，非 ratio 缺口）。"""
    R = g["R"]; Vfinal = g["Vfinal"]; init_affine = g["init_affine"]
    probe_idx = g["probe_idx"]
    Vstar, _, _ = value_iteration(gamma=GAMMA, tol=1e-5)
    vstar_t = torch.tensor([Vstar[R[i]] for i in probe_idx])
    init_t = init_affine[probe_idx]
    final_t = Vfinal[probe_idx]
    rms_star_init = float(((vstar_t - init_t) ** 2).mean() ** 0.5)
    tau_disp = 0.3 * rms_star_init
    disp = float(((final_t - init_t) ** 2).mean() ** 0.5)
    disp_pass = disp > tau_disp
    # corr(V_net, heuristic4) over reachable（advisory）
    vnet_all = Vfinal.detach()
    h_all = torch.tensor([heuristic4(b) for b in R], dtype=torch.float32)
    vm = vnet_all.mean(); hm = h_all.mean()
    cov = ((vnet_all - vm) * (h_all - hm)).mean()
    corr = float(cov / (vnet_all.std(unbiased=False) * h_all.std(unbiased=False) + 1e-9))
    corr_flag = corr >= 0.92      # 触发 USER_GATE 人工核查（不自动 FAIL）
    print(f"[位移检查] disp(V-unit RMS)={disp:.4f}  τ_disp={tau_disp:.4f}"
          f"(=0.3×RMS(V*−init)={rms_star_init:.4f})  binding → "
          f"{'PASS' if disp_pass else 'FAIL'}", flush=True)
    print(f"[位移检查] corr(V_net,heuristic4)={corr:.4f}  advisory 阈<0.92  → "
          f"{'OK' if not corr_flag else 'USER_GATE 人工核查'}", flush=True)
    return {"disp_pass": bool(disp_pass), "disp_v_unit": disp, "tau_disp": tau_disp,
            "rms_star_init": rms_star_init, "corr_heuristic": corr,
            "corr_user_gate_flag": bool(corr_flag)}


# ======================================================================
# main
# ======================================================================
if __name__ == "__main__":
    _test_identity_transform()
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"
    print(f"=== rl4.py | torch {torch.__version__} | mode={mode} ===", flush=True)
    # 合并进已有 JSON（分别跑 gamma / T 时不互相覆盖）
    try:
        results = json.load(open("rl4_gate.json"))
    except (FileNotFoundError, json.JSONDecodeError):
        results = {}

    if mode in ("gamma", "all"):
        g = train_gamma()
        gg = gate_gamma(g)
        results["gamma_gate"] = gg
        results["gamma_train"] = {k: g[k] for k in
                                  ("v_empty", "sweeps", "wall_s", "warm_used",
                                   "plateau", "wall_budget_exceeded")}
        gamma_pass = gg["value_pass"] and gg["policy_pass"]
        results["gamma_pass"] = gamma_pass
        # 位移检查仅在 γ-gate(值+策略) 已 PASS 后评估（条件 gate）
        if gamma_pass:
            results["displacement"] = displacement_check(g)

    if mode in ("T", "all"):
        t = train_T()
        tg = gate_T(t)
        results["T_gate"] = tg
        results["T_train"] = {k: t[k] for k in
                              ("v8", "v16", "sweeps", "wall_s", "warm_used",
                               "plateau", "wall_budget_exceeded")}
        results["T_pass"] = tg["value_pass"] and tg["online_pass"]

    json.dump(results, open("rl4_gate.json", "w"), ensure_ascii=False, indent=2)
    print("\n-> wrote rl4_gate.json", flush=True)
    print(json.dumps(results, ensure_ascii=False, indent=2), flush=True)
