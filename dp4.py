"""4×4 精确 DP「可解类比」——经三轮独立审核（APPROVE）。

目的：在一个**可精确求解**的小棋盘上，拿到 8×8 拿不到的两样东西——
  (a) 真·最优(value iteration) → 量化"近视贪心启发式离真最优多远"；
  (b) 精确的信息价值(offline−online)。
这是**类比**，不是 8×8 数字的标定。三处简化（README 头部声明）：
  ① 每回合发 1 块（非 3 块手牌）→ MDP 状态=棋盘(2^16)；
  ② 线性计分=消除行列数，无 combo/all-clear（combo 会进状态、爆炸）；
  ③ γ-折扣(γ=0.99)：否则 monomino 永远放得下→无界存活→V* 发散(审核2 抓出)。
故度量是**discounted VOI**，且不建模 8×8 的 combo 运气 / 3 块重排技巧。

折扣约定（审核3 锁定）：从空盘第 t 步(0-indexed)的即时奖励权重 γ^t，
online V*/π*-模拟/offline DP **三者完全一致**，都度量"从空盘起的折扣回报"。
online 与 offline 在**同一截断长度 L** 上比较(γ^L·maxR/(1−γ)<ε)，保证可比。
"""

import random
from pieces import CATALOG

N = 4
NCELL = N * N
FULL = (1 << NCELL) - 1
ROW = [sum(1 << (r * N + c) for c in range(N)) for r in range(N)]
COL = [sum(1 << (r * N + c) for r in range(N)) for c in range(N)]


def _fit_placements(cells):
    """piece 在 4×4 的所有合法落点 mask（bounding box ≤4，cells≤4 已在筛选时保证）。"""
    maxr = max(r for r, _ in cells); maxc = max(c for _, c in cells)
    masks = []
    for r0 in range(N - maxr):
        for c0 in range(N - maxc):
            m = 0
            for r, c in cells:
                m |= 1 << ((r0 + r) * N + (c0 + c))
            masks.append(m)
    return masks


# 筛选 ≤4 cells 且能放进 4×4 的 piece
CATALOG4 = [(nm, cells) for (nm, cells) in CATALOG
            if len(cells) <= 4 and max(r for r, _ in cells) <= N - 1
            and max(c for _, c in cells) <= N - 1]
PIECES4 = [cells for _, cells in CATALOG4]
K = len(PIECES4)
PLACE4 = [_fit_placements(c) for c in PIECES4]


def apply4(board, mask):
    """放置(已确认合法) + 单遍消除(同时算所有满行满列,清一次,无级联)。
    返回 (新盘, 消除行列数)。与 fast.apply_mask 语义一致。"""
    b = board | mask
    clr = 0; cleared = 0
    for r in range(N):
        if b & ROW[r] == ROW[r]:
            clr |= ROW[r]; cleared += 1
    for c in range(N):
        if b & COL[c] == COL[c]:
            clr |= COL[c]; cleared += 1
    return b & ~clr, cleared


def heuristic4(board):
    """4×4 启发式(只给 greedy 当 tie-break，常数未对 4×4 标定，已声明)：
    -填充数 - 0.5*碎片度(空格四邻被墙/填充挡住的方向数)。"""
    filled = bin(board).count("1")
    frag = 0
    for i in range(NCELL):
        if board & (1 << i):
            continue
        r, c = divmod(i, N)
        for dr, dc in ((0, 1), (0, -1), (1, 0), (-1, 0)):
            nr, nc = r + dr, c + dc
            if not (0 <= nr < N and 0 <= nc < N) or (board & (1 << (nr * N + nc))):
                frag += 1
    return -1.0 * filled - 0.5 * frag


# 预计算转移表：MOVES[piece][board] 太大(K×2^16)，改为按需 + 缓存合法落点。
# 单步：给定 board, piece → 枚举 PLACE4[piece] 中不冲突的 mask → (board',cleared)
def legal_moves(board, pid):
    out = []
    for m in PLACE4[pid]:
        if not (board & m):
            out.append(apply4(board, m))
    return out


def reachable_boards():
    """从空盘 BFS：所有可达的"静息棋盘"(放置+消除后的状态)。多数 2^16 不可达。"""
    seen = {0}
    frontier = [0]
    while frontier:
        nxt = []
        for b in frontier:
            for p in range(K):
                for (nb, _cl) in legal_moves(b, p):
                    if nb not in seen:
                        seen.add(nb); nxt.append(nb)
        frontier = nxt
    return seen


def value_iteration(gamma=0.99, tol=1e-3, max_sweeps=6000):
    """γ-折扣无限期 VI，仅在可达棋盘上：V[b]=mean_p[ max_pos(cleared+γV[b']) ; 0 ]。
    γ-收缩 → 唯一有限 V*，单调收敛。返回 (V dict, sweeps, |reachable|)。"""
    R = reachable_boards()
    moves = {b: [legal_moves(b, p) for p in range(K)] for b in R}
    V = {b: 0.0 for b in R}
    for sweep in range(max_sweeps):
        delta = 0.0
        for b in R:
            tot = 0.0; mb = moves[b]
            for p in range(K):
                best = 0.0
                for (nb, cl) in mb[p]:
                    v = cl + gamma * V[nb]
                    if v > best:
                        best = v
                tot += best
            tot /= K
            d = abs(tot - V[b])
            if d > delta:
                delta = d
            V[b] = tot
        if delta < tol:
            break
    return V, (sweep + 1), len(R)


def play_online(V, seq, gamma, L):
    """模拟 π*(=对 V* 贪心) 在给定序列上，折扣回报，截断 L。"""
    b = 0; ret = 0.0; disc = 1.0
    for t in range(min(L, len(seq))):
        p = seq[t]
        best = None
        for m in PLACE4[p]:
            if b & m:
                continue
            nb, cl = apply4(b, m)
            v = cl + gamma * V[nb]
            if best is None or v > best[0]:
                best = (v, nb, cl)
        if best is None:
            break  # 放不下 → 卡死
        ret += disc * best[2]; b = best[1]; disc *= gamma
    return ret


def play_greedy(seq, gamma, L):
    """近视贪心(单块 argmax cleared+heuristic)；折扣回报，截断 L。"""
    b = 0; ret = 0.0; disc = 1.0
    for t in range(min(L, len(seq))):
        p = seq[t]
        best = None
        for m in PLACE4[p]:
            if b & m:
                continue
            nb, cl = apply4(b, m)
            v = cl + heuristic4(nb)
            if best is None or v > best[0]:
                best = (v, nb, cl)
        if best is None:
            break
        ret += disc * best[2]; b = best[1]; disc *= gamma
    return ret


def horizon_for(gamma, eps=1e-3, maxR=8.0):
    """选截断 L 使 γ^L·maxR/(1−γ) < eps（远未来折扣可忽略，offline/online 同 L）。"""
    import math
    return int(math.ceil(math.log(eps * (1 - gamma) / maxR) / math.log(gamma)))


def run(M=200, gamma=0.95, seed0=0):
    """主实验：VI→V*；M 条序列上 online(π*)/offline(DP)/greedy 同 L、CRN 配对；
    discounted VOI + 启发式缺口 + bootstrap CI。返回结果 dict。"""
    from oracle_analysis import bootstrap_ci, pct
    L = horizon_for(gamma)
    V, sw, nR = value_iteration(gamma=gamma)
    on, off, gr = [], [], []
    for s in range(seed0, seed0 + M):
        rng = random.Random(f"dp4-{s}")
        seq = [rng.randrange(K) for _ in range(L)]
        on.append(play_online(V, seq, gamma, L))
        off.append(offline_optimal(seq, gamma, L))
        gr.append(play_greedy(seq, gamma, L))
    voi = [o - n for o, n in zip(off, on)]       # discounted 信息价值
    hgap = [n - g for n, g in zip(on, gr)]        # 最优 − 近视贪心
    mean_on = sum(on) / M
    se_on = (sum((x - mean_on) ** 2 for x in on) / M / M) ** 0.5
    out = {"gamma": gamma, "L": L, "M": M, "sweeps": sw, "reachable": nR,
           "V_star_empty": V[0], "mean_online": mean_on,
           "assert_online_eq_Vstar": abs(mean_on - V[0]) < 3 * se_on + 1e-6,
           "online_se": se_on}
    vm, vlo, vhi = bootstrap_ci(voi, boot_seed="dp4-voi")
    vmm = pct(voi, 50)
    hm, hlo, hhi = bootstrap_ci(hgap, boot_seed="dp4-hgap")
    out["discounted_VOI"] = {"mean": vm, "ci": [vlo, vhi], "median": vmm}
    out["heuristic_gap"] = {"mean": hm, "ci": [hlo, hhi]}
    out["greedy_optimality_ratio"] = (sum(gr) / M) / V[0] if V[0] else float("nan")
    out["means"] = {"online_opt": mean_on, "offline_opt": sum(off) / M,
                    "greedy": sum(gr) / M}
    return out


def offline_optimal(seq, gamma, L):
    """离线最优：已知整条序列，前向 DP 求最大折扣回报(截断 L)。
    f[b] = 到达 board b 的最大"已累计折扣回报"。每步只放 seq[t] 一块。
    奖励非负 → 任意时刻可'收手' → offline = 过程中见过的最大 f。"""
    seq = seq[:L]
    f = {0: 0.0}
    best_ret = 0.0
    disc = 1.0
    for t, p in enumerate(seq):
        nf = {}
        for b, acc in f.items():
            placed = False
            for m in PLACE4[p]:
                if b & m:
                    continue
                placed = True
                nb, cl = apply4(b, m)
                val = acc + disc * cl
                if nb not in nf or val > nf[nb]:
                    nf[nb] = val
                if val > best_ret:
                    best_ret = val
            # 放不下：该路径在此卡死，acc 已计入 best_ret(历史最大)
        if not nf:
            break
        f = nf
        disc *= gamma
    return best_ret


if __name__ == "__main__":
    import sys, json
    M = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    print(f"4×4 exact-DP analog | K={K} pieces | M={M}", flush=True)
    # 主结果 γ=0.95；并报 V* 在 γ∈{0.9,0.95,0.99} 的敏感性(VI 便宜)
    res = run(M=M, gamma=0.95)
    print(json.dumps(res, ensure_ascii=False, indent=2), flush=True)
    sens = {}
    for g in (0.90, 0.95, 0.99):
        V, sw, nR = value_iteration(gamma=g)
        sens[g] = {"V_star_empty": V[0], "sweeps": sw}
        print(f"  V*(empty) @ γ={g}: {V[0]:.3f} ({sw} sweeps)", flush=True)
    res["V_star_sensitivity"] = {str(k): v for k, v in sens.items()}
    json.dump(res, open("dp4.json", "w"), ensure_ascii=False, indent=2)
    print("-> wrote dp4.json", flush=True)
