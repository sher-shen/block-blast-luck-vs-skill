"""CEM-Dellacherie 玩家 + 训练器（目标线①：最强可玩策略）。

torch-free（守"torch 只在 rl8/rl4"硬约定，本文件只 import fast/scoring/random）。
思路 = Tetris SOTA：手工特征线性评估 + CEM(交叉熵)黑盒优化权重直接对**真实实战得分**调。
关键好处：完全不自举(no bootstrap) -> 结构上免疫 CNN-V 的 OOD 高估早死病根。

特征(8 维, 全位运算; 用于 argmax 末态 -> 不要 bias/常数项):
  filled      已填格数            (penalize)
  frag        碎片度(空格邻接墙/填)  (penalize)
  holes       四邻全堵的空格(只能1x1) (penalize)
  row_trans   行 填<->空 转换数      (penalize; Dellacherie)
  col_trans   列 填<->空 转换数      (penalize; Dellacherie)
  empty_2x2   全空 2x2 块数(大块落脚)  (reward)
  empty_3x3   全空 3x3 块数          (reward)
  near_full   差一格就消的行/列数     (sign 让 CEM 学)

CLI:
  python cem.py smoke
  python cem.py train [gens=25] [pop=64] [elite=12] [ngames=24] [B_train=0]   # B_train=0 -> greedy 训练
  python cem.py bench [M=200] [T=50] [B=12]
  python cem.py play [seed=0] [B=12]
"""
import json
import math
import random
import sys
import time

import fast
from fast import (FULL, COL, ROW, COL0, COL7, ROW0, ROW7,
                  NOTC7, NOTC0, NOTR7, NOTR0, PLACE, NCELLS, NUM_TYPES,
                  apply_mask, SCORING)
from scoring import score_placement

# 3x3 全空锚点掩码：anchor (r,c) 需 r<=5 且 c<=5
ANCHOR3 = sum(1 << (r * 8 + c) for r in range(6) for c in range(6))

NFEAT = 8
FEAT_NAMES = ["filled", "frag", "holes", "row_trans", "col_trans",
              "empty_2x2", "empty_3x3", "near_full"]
# 量纲缩放：让每维 ~O(1)，权重可比、CEM 的 sigma=1 有意义
SCALE = [1.0 / 64, 1.0 / 100, 1.0 / 32, 1.0 / 72, 1.0 / 72,
         1.0 / 49, 1.0 / 36, 1.0 / 16]


def features(board):
    """返回长度 NFEAT 的已缩放特征 list（越大方向由权重决定）。纯位运算。"""
    filled = board.bit_count()
    empty = (~board) & FULL

    # 四方向：邻居为(填充 或 墙) -> 对该空格置位（= 该方向被堵）
    right = ((board >> 1) & NOTC7) | COL7
    left = ((board << 1) & NOTC0 & FULL) | COL0
    down = ((board >> 8) & NOTR7) | ROW7
    up = ((board << 8) & NOTR0 & FULL) | ROW0

    frag = ((empty & right).bit_count() + (empty & left).bit_count()
            + (empty & down).bit_count() + (empty & up).bit_count())
    holes = (empty & right & left & down & up).bit_count()

    # 行/列转换（墙算填充）
    h_int = ((board ^ (board >> 1)) & NOTC7).bit_count()
    row_trans = h_int + (empty & COL0).bit_count() + (empty & COL7).bit_count()
    v_int = ((board ^ (board >> 8)) & NOTR7).bit_count()
    col_trans = v_int + (empty & ROW0).bit_count() + (empty & ROW7).bit_count()

    # 大块空位
    e = empty
    two = e & (e >> 1) & (e >> 8) & (e >> 9) & NOTC7 & NOTR7
    empty_2x2 = two.bit_count()
    three = (e & (e >> 1) & (e >> 2) & (e >> 8) & (e >> 9) & (e >> 10)
             & (e >> 16) & (e >> 17) & (e >> 18) & ANCHOR3)
    empty_3x3 = three.bit_count()

    # 差一格就消的行/列
    near = 0
    for r in range(8):
        if (board & ROW[r]).bit_count() == 7:
            near += 1
    for c in range(8):
        if (board & COL[c]).bit_count() == 7:
            near += 1

    raw = (filled, frag, holes, row_trans, col_trans,
           empty_2x2, empty_3x3, near)
    return [raw[i] * SCALE[i] for i in range(NFEAT)]


def linval(board, w):
    f = features(board)
    return sum(w[i] * f[i] for i in range(NFEAT))


# ---------- 玩家 ----------
def cem_greedy_hand(board, combo, hand, w):
    """按发牌顺序逐块选 pts + w·feat(末态) 最大的落点（1-ply，训练用，廉价）。"""
    total = 0
    for pid in hand:
        best = None
        for m in PLACE[pid]:
            if board & m:
                continue
            nb, cl, empty = apply_mask(board, m)
            pts, nc = score_placement(SCORING, NCELLS[pid], cl, empty, combo)
            val = pts + linval(nb, w)
            if best is None or val > best[0]:
                best = (val, nb, nc, pts)
        if best is None:
            return total, board, combo, False
        _, board, combo, pts = best
        total += pts
    return total, board, combo, True


def cem_hand(board, combo, hand, w, B=12):
    """联合 beam(顺序×落点, 连击贯穿)，排序用 pts + w·feat。比 1-ply 更强，最终评估用。"""
    states = [(board, combo, 0, ())]
    for _ in range(3):
        cand = []
        for (b, c, sc, used) in states:
            for i in range(3):
                if i in used:
                    continue
                pid = hand[i]
                for m in PLACE[pid]:
                    if b & m:
                        continue
                    nb, cl, empty = apply_mask(b, m)
                    pts, nc = score_placement(SCORING, NCELLS[pid], cl, empty, c)
                    cand.append((nb, nc, sc + pts, used + (i,)))
        if not cand:
            return 0, board, combo, False
        cand.sort(key=lambda s: s[2] + linval(s[0], w), reverse=True)
        states = cand[:B]
    best = max(states, key=lambda s: s[2] + linval(s[0], w))
    return best[2], best[0], best[1], True


def play_cem(stream, w, B=0):
    """玩一条 CRN 流。B=0 -> 1-ply greedy(快, 训练用); B>0 -> beam(强)。返回 (total, surv)。"""
    board, combo, total = 0, 0, 0.0
    for rnd, hand in enumerate(stream):
        if B > 0:
            sc, board, combo, alive = cem_hand(board, combo, hand, w, B)
        else:
            sc, board, combo, alive = cem_greedy_hand(board, combo, hand, w)
        if not alive:
            return total, rnd
        total += sc
    return total, len(stream)


# ---------- Phase 2: CEM 评估塞进前瞻搜索 ----------
def cem_beam_cands(board, combo, hand, w, B=12):
    """top-B 末态候选(按 linval 剪枝)。返回 [(nb,nc,sc),...]。空=死。"""
    states = [(board, combo, 0, ())]
    for _ in range(3):
        cand = []
        for (b, c, sc, used) in states:
            for i in range(3):
                if i in used:
                    continue
                pid = hand[i]
                for m in PLACE[pid]:
                    if b & m:
                        continue
                    nb, cl, empty = apply_mask(b, m)
                    pts, nc = score_placement(SCORING, NCELLS[pid], cl, empty, c)
                    cand.append((nb, nc, sc + pts, used + (i,)))
        if not cand:
            return []
        cand.sort(key=lambda s: s[2] + linval(s[0], w), reverse=True)
        states = cand[:B]
    return [(b, c, sc) for (b, c, sc, _) in states]


def _beam_hand_one(board, combo, hand, w, Br=3, key="cem"):
    """用联合 beam(宽 Br, 连击贯穿)铺完整手, 取最优末态。
    key='cem' -> 排序键 pts + linval(w)（off-policy w）; 'heur' -> pts + fast.heuristic（无 off-policy）。
    返回 (加分, 末盘, 末连击, alive)。复用 cem_hand/beam_hand 结构, 但按需 Br 宽、可换排序键。"""
    n = len(hand)
    states = [(board, combo, 0, ())]
    for _ in range(n):
        cand = []
        for (b, c, sc, used) in states:
            for i in range(n):
                if i in used:
                    continue
                pid = hand[i]
                for m in PLACE[pid]:
                    if b & m:
                        continue
                    nb, cl, empty = apply_mask(b, m)
                    pts, nc = score_placement(SCORING, NCELLS[pid], cl, empty, c)
                    cand.append((nb, nc, sc + pts, used + (i,)))
        if not cand:
            return 0, board, combo, False
        if key == "heur":
            cand.sort(key=lambda s: s[2] + fast.heuristic(s[0]), reverse=True)
        else:
            cand.sort(key=lambda s: s[2] + linval(s[0], w), reverse=True)
        states = cand[:Br]
    if key == "heur":
        best = max(states, key=lambda s: s[2] + fast.heuristic(s[0]))
    else:
        best = max(states, key=lambda s: s[2] + linval(s[0], w))
    return best[2], best[0], best[1], True


def _rollout_leaf(board, combo, future, w, base="heur", Br=3):
    """按给定牌流滚动 D 手(base 策略选落点)，返回 (累计分, 末盘, 末连击)。
    base in {heur,cem}: 逐块 1-ply（原行为，不动，防回归）。
    base in {beam,hbeam}: 每手用联合 beam（宽 Br）铺整手（beam=linval/w 排序, hbeam=heuristic 排序）。"""
    total = 0
    if base in ("beam", "hbeam"):
        key = "heur" if base == "hbeam" else "cem"
        for i in range(0, len(future), 3):
            sc, board, combo, alive = _beam_hand_one(board, combo, future[i:i + 3],
                                                     w, Br, key)
            if not alive:
                return total, board, combo
            total += sc
        return total, board, combo
    for i in range(0, len(future), 3):
        for pid in future[i:i + 3]:
            best = None
            for m in PLACE[pid]:
                if board & m:
                    continue
                nb, cl, empty = apply_mask(board, m)
                pts, ncb = score_placement(SCORING, NCELLS[pid], cl, empty, combo)
                key = pts + (fast.heuristic(nb) if base == "heur" else linval(nb, w))
                if best is None or key > best[0]:
                    best = (key, nb, ncb, pts)
            if best is None:
                return total, board, combo
            _, board, combo, pts = best
            total += pts
    return total, board, combo


def play_cem_look(stream, w, D=2, S=20, B=12, base="heur", cand="cem", seed_idx=0,
                  Br=3, use_leaf=True):
    """CEM 价值引导前瞻：beam 候选 -> S 条 D 手 rollout + linval 叶尾值 -> 选最优。
    cand: 'cem'=按 linval 剪枝候选, 'heur'=按 heuristic(=fast.beam_hand)。
    base: rollout 基策略选点用 'heur'/'cem'(1-ply) 或 'beam'/'hbeam'(宽 Br 联合 beam 铺整手)。
    Br: beam/hbeam 基的手内 beam 宽。
    use_leaf: True=留 linval 叶尾值(变体A); False=砍叶V, 仅 rollout 累计分(变体B, 配深 D 吃尾)。
    rollout RNG 由 (seed_idx,move) 定 -> CRN 不破。"""
    board, combo, total = 0, 0, 0.0
    for move, hand in enumerate(stream):
        cands = (cem_beam_cands(board, combo, hand, w, B) if cand == "cem"
                 else fast.beam_hand(board, combo, hand, B))
        if not cands:
            return total, move
        fstreams = [[random.Random(f"rollout-{seed_idx}-{move}-{s}").randrange(NUM_TYPES)
                     for _ in range(3 * D)] for s in range(S)]
        best = None
        for (nb, nc, hs) in cands:
            acc = 0.0
            for fs in fstreams:
                rt, eb, ec = _rollout_leaf(nb, nc, fs, w, base, Br)
                acc += rt + (linval(eb, w) if use_leaf else 0.0)
            value = hs + acc / S
            if best is None or value > best[0]:
                best = (value, hs, nb, nc)
        total += best[1]
        board = best[2]
        combo = best[3]
    return total, len(stream)


def _look_worker(args):
    st, w, D, S, B, base, cand, i, Br, use_leaf = args
    return play_cem_look(st, w, D, S, B, base, cand, i, Br, use_leaf)


def bench_look(M=100, T=50, grid=None, wfile="models/cem_w.json", parallel=True):
    """并行擂台：CEM 前瞻 vs 交付基线(look 2842 / vla 2950 由 rl8 报)。"""
    data = json.load(open(wfile, encoding="utf-8"))
    w = data.get("best_w") or data["mu"]
    streams = gen_streams("bench", M, T)
    if grid is None:
        grid = [(2, 20, 12, "heur", "cem"), (2, 30, 12, "heur", "cem"),
                (3, 20, 12, "heur", "cem"), (2, 30, 12, "cem", "cem")]
    print(f"=== CEM 前瞻擂台 | M={M} bench-CRN | T={T} | 不预知未来 ===", flush=True)
    pool = None
    if parallel:
        from concurrent.futures import ProcessPoolExecutor
        pool = ProcessPoolExecutor()
    res = {}
    for entry in grid:
        # grid 项可为 (D,S,B,base,cand) / +Br / +Br,use_leaf
        D, S, B, base, cand = entry[:5]
        Br = entry[5] if len(entry) > 5 else 3
        use_leaf = entry[6] if len(entry) > 6 else True
        extra = (f" Br{Br}" if base in ("beam", "hbeam") else "") + ("" if use_leaf else " noleaf")
        label = f"vla D{D} S{S} B{B} base={base} cand={cand}{extra}"
        t0 = time.time()
        argz = [(streams[i], w, D, S, B, base, cand, i, Br, use_leaf) for i in range(M)]
        rows = list(pool.map(_look_worker, argz)) if pool else [_look_worker(a) for a in argz]
        totals = [r[0] for r in rows]
        survs = [float(r[1]) for r in rows]
        mean, se = _block_stats(totals)
        smean, _ = _block_stats(survs)
        res[label] = {"mean": mean, "se": se, "surv": smean}
        print(f"[{label:34s}] mean={mean:7.1f} +/- {se:5.1f}  surv={smean:5.1f}"
              f"  ({time.time()-t0:.0f}s)", flush=True)
    if pool:
        pool.shutdown()
    json.dump(res, open("cem_look.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print("-> wrote cem_look.json", flush=True)
    return res


# ---------- CRN 流 ----------
def gen_streams(tag, M, T):
    out = []
    for m in range(M):
        rng = random.Random(f"{tag}-{m}")
        out.append([[rng.randrange(NUM_TYPES) for _ in range(3)] for _ in range(T)])
    return out


# ---------- CEM 训练 ----------
def _eval_candidate(args):
    """worker：一个权重在一批流上的平均实战分。top-level 以便 multiprocessing 可 pickle。"""
    w, streams, B = args
    s = 0.0
    for st in streams:
        s += play_cem(st, w, B)[0]
    return s / len(streams)


def cem_train(gens=25, pop=64, elite=12, ngames=24, T=50, B=0,
              seed="cem", parallel=True, out="models/cem_w.json"):
    """CEM：每代采样 pop 个权重，在共享 CRN 流(代内配对)上评分，取 elite 重拟 mu/sigma。
    训练流 tag=cem-train-{gen}（每代换流防过拟合；与报告用的 bench tag 分离=防泄漏）。"""
    d = NFEAT
    mu = [0.0] * d
    sigma = [1.0] * d
    best_w, best_fit = None, -1e18
    pool = None
    if parallel:
        try:
            from concurrent.futures import ProcessPoolExecutor
            pool = ProcessPoolExecutor()
        except Exception as ex:
            print(f"[warn] no pool ({ex}); single-proc", flush=True)
            pool = None

    hist = []
    for g in range(gens):
        t0 = time.time()
        streams = gen_streams(f"cem-train-{g}", ngames, T)
        rng = random.Random(f"{seed}-{g}")
        cands = [[mu[i] + sigma[i] * rng.gauss(0, 1) for i in range(d)]
                 for _ in range(pop)]
        argz = [(w, streams, B) for w in cands]
        if pool is not None:
            fits = list(pool.map(_eval_candidate, argz))
        else:
            fits = [_eval_candidate(a) for a in argz]

        order = sorted(range(pop), key=lambda i: fits[i], reverse=True)
        elites = [cands[i] for i in order[:elite]]
        # 重拟
        for i in range(d):
            col = [e[i] for e in elites]
            m = sum(col) / elite
            var = sum((x - m) ** 2 for x in col) / elite
            noise = max(0.0, 0.10 - 0.004 * g)   # Szita-Lorincz 噪声地板，随代衰减
            mu[i] = m
            sigma[i] = math.sqrt(var) + noise

        gen_best_fit = fits[order[0]]
        gen_best_w = cands[order[0]]
        if gen_best_fit > best_fit:
            best_fit, best_w = gen_best_fit, list(gen_best_w)
        hist.append({"gen": g, "best": gen_best_fit,
                     "elite_mean": sum(fits[i] for i in order[:elite]) / elite})
        print(f"[gen {g:2d}] best={gen_best_fit:7.1f}  elite_mean="
              f"{hist[-1]['elite_mean']:7.1f}  ({time.time()-t0:.0f}s)", flush=True)

    if pool is not None:
        pool.shutdown()

    result = {"mu": mu, "best_w": best_w, "best_fit": best_fit,
              "feat_names": FEAT_NAMES, "scale": SCALE,
              "cfg": {"gens": gens, "pop": pop, "elite": elite,
                      "ngames": ngames, "T": T, "B_train": B},
              "hist": hist}
    import os
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(result, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"-> wrote {out}", flush=True)
    print("-> mu:", "  ".join(f"{FEAT_NAMES[i]}={mu[i]:+.3f}" for i in range(d)), flush=True)
    return result


# ---------- 评测 ----------
def _block_stats(xs, nb=16):
    M = len(xs)
    blk = M // nb
    bm = [sum(xs[i * blk:(i + 1) * blk]) / blk for i in range(nb)]
    mean = sum(bm) / nb
    se = (sum((x - mean) ** 2 for x in bm) / (nb - 1) / nb) ** 0.5
    return mean, se


def _paired_ci(diffs, n_boot=2000, seed="cem-boot"):
    M = len(diffs)
    rng = random.Random(seed)
    boots = []
    for _ in range(n_boot):
        s = sum(diffs[rng.randrange(M)] for _ in range(M))
        boots.append(s / M)
    boots.sort()
    return boots[int(0.025 * n_boot)], boots[int(0.975 * n_boot)]


def bench(M=200, T=50, B=12, wfile="models/cem_w.json"):
    data = json.load(open(wfile, encoding="utf-8"))
    w = data.get("best_w") or data["mu"]
    streams = gen_streams("bench", M, T)   # 与训练流(cem-train-*)分离的 held-out
    print(f"=== CEM 擂台 | M={M} CRN(bench tag) | T={T} | B={B} | 不预知未来 ===", flush=True)

    def run(label, fn):
        t0 = time.time()
        rows = [fn(streams[i]) for i in range(M)]
        totals = [r[0] for r in rows]
        survs = [float(r[1]) for r in rows]
        mean, se = _block_stats(totals)
        smean, _ = _block_stats(survs)
        print(f"[{label:22s}] mean={mean:7.1f} +/- {se:5.1f}  surv={smean:5.1f}/{T}"
              f"  ({time.time()-t0:.0f}s)", flush=True)
        return totals, mean

    out = {}
    out["cem_beam"], m_cem = run(f"cem beam B{B}", lambda s: play_cem(s, w, B))
    out["cem_greedy"], _ = run("cem greedy 1-ply", lambda s: play_cem(s, w, 0))
    out["greedy"], _ = run("greedy(heur)", lambda s: _play_greedy(s))
    out["strong"], _ = run("strong B12", lambda s: fast.play_strong(s, B=12)
                           if hasattr(fast, "play_strong") else _play_strong(s, 12))

    base = out["strong"]
    print("\n=== 相对 strong B12 配对差(>0=更强, 禁 ratio) ===", flush=True)
    for label in ("cem_beam", "cem_greedy", "greedy"):
        diffs = [out[label][i] - base[i] for i in range(M)]
        dm, dse = _block_stats(diffs)
        lo, hi = _paired_ci(diffs)
        flag = "STRONGER" if lo > 0 else ("WEAKER" if hi < 0 else "~tie")
        print(f"  {label:14s} d={dm:+7.1f} +/- {dse:5.1f}  95%CI=[{lo:+.1f},{hi:+.1f}]  {flag}",
              flush=True)
    json.dump({"M": M, "T": T, "B": B, "cem_beam_mean": m_cem},
              open("cem_bench.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print("-> wrote cem_bench.json", flush=True)


def _play_greedy(stream):
    board, combo, total = 0, 0, 0.0
    for rnd, hand in enumerate(stream):
        sc, board, combo, alive = fast.greedy_hand(board, combo, hand)
        if not alive:
            return total, rnd
        total += sc
    return total, len(stream)


def _play_strong(stream, B):
    board, combo, total = 0, 0, 0.0
    for rnd, hand in enumerate(stream):
        sc, board, combo, alive = fast.strong_hand(board, combo, hand, B=B)
        if not alive:
            return total, rnd
        total += sc
    return total, len(stream)


def perseed(D=2, S=30, B=12, base="heur", cand="cem", M=200, T=50,
            out="C:/tmp/cem_perseed.json", wfile="models/cem_w.json"):
    """并行 dump CEM 前瞻每-seed 实战分(bench tag) → 供与 CNN-vla 配对。"""
    data = json.load(open(wfile, encoding="utf-8"))
    w = data.get("best_w") or data["mu"]
    streams = gen_streams("bench", M, T)
    from concurrent.futures import ProcessPoolExecutor
    argz = [(streams[i], w, D, S, B, base, cand, i, 3, True) for i in range(M)]
    with ProcessPoolExecutor() as pool:
        rows = list(pool.map(_look_worker, argz))
    tot = [r[0] for r in rows]
    json.dump(tot, open(out, "w", encoding="utf-8"))
    print(f"CEM vla D{D} S{S} M={M} mean={sum(tot)/M:.1f} -> {out}", flush=True)


def paircfg(D=2, S=6, B=12, base="beam", cand="cem", Br=3, use_leaf=True,
            cD=2, cS=30, cB=12, cbase="heur", ccand="cem",
            M=50, T=50, wfile="models/cem_w.json"):
    """预算中性配对裁决：处理配置 vs 对照(默认 D2 S30 base=heur)在**同一 bench CRN 流**上跑，
    给 per-seed 配对差 d ± 16-block-SE + bootstrap 95%CI(禁 ratio)。这是 Step1/Step2 的决定性裁判。"""
    data = json.load(open(wfile, encoding="utf-8"))
    w = data.get("best_w") or data["mu"]
    streams = gen_streams("bench", M, T)
    from concurrent.futures import ProcessPoolExecutor
    c_argz = [(streams[i], w, cD, cS, cB, cbase, ccand, i, 3, True) for i in range(M)]
    t_argz = [(streams[i], w, D, S, B, base, cand, i, Br, use_leaf) for i in range(M)]
    tl = (f"vla D{D} S{S} B{B} base={base} cand={cand}"
          + (f" Br{Br}" if base in ("beam", "hbeam") else "")
          + ("" if use_leaf else " noleaf"))
    cl = f"vla D{cD} S{cS} B{cB} base={cbase} cand={ccand}"
    print(f"=== 预算中性配对 | M={M} 同 bench-CRN | T={T} ===", flush=True)
    print(f"  treat = {tl}", flush=True)
    print(f"  ctrl  = {cl}", flush=True)
    t0 = time.time()
    with ProcessPoolExecutor() as pool:
        crows = list(pool.map(_look_worker, c_argz))
        trows = list(pool.map(_look_worker, t_argz))
    ctrl = [r[0] for r in crows]
    trt = [r[0] for r in trows]
    cm, cse = _block_stats(ctrl)
    tm, tse = _block_stats(trt)
    csv = sum(float(r[1]) for r in crows) / M
    tsv = sum(float(r[1]) for r in trows) / M
    diffs = [trt[i] - ctrl[i] for i in range(M)]
    dm, dse = _block_stats(diffs)
    lo, hi = _paired_ci(diffs)
    flag = ("TREAT STRONGER" if lo > 0 else ("TREAT WEAKER" if hi < 0 else "TIE (CI crosses 0)"))
    print(f"  ctrl  : {cm:7.1f} +/- {cse:5.1f}  surv={csv:5.1f}", flush=True)
    print(f"  treat : {tm:7.1f} +/- {tse:5.1f}  surv={tsv:5.1f}", flush=True)
    print(f"  paired d(treat-ctrl) = {dm:+7.1f} +/- {dse:5.1f}  95%CI=[{lo:+.1f},{hi:+.1f}]  {flag}"
          f"  ({time.time()-t0:.0f}s)", flush=True)
    out = {"treat": tl, "ctrl": cl, "M": M, "T": T,
           "treat_mean": tm, "treat_se": tse, "ctrl_mean": cm, "ctrl_se": cse,
           "d": dm, "d_se": dse, "ci": [lo, hi], "flag": flag,
           "treat_per_seed": trt, "ctrl_per_seed": ctrl}
    fn = f"cem_paircfg_{base}{'_noleaf' if not use_leaf else ''}_D{D}S{S}Br{Br}.json"
    json.dump(out, open(fn, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"-> wrote {fn}", flush=True)
    return out


def pair(cemf="C:/tmp/cem_perseed.json", cnnf="C:/tmp/cnn_perseed.json"):
    """torch-free 配对：CEM vs CNN-vla 同流每-seed 差 + 16-block-SE + bootstrap CI(禁 ratio)。"""
    a = json.load(open(cemf, encoding="utf-8"))
    b = json.load(open(cnnf, encoding="utf-8"))
    M = min(len(a), len(b))
    am, ase = _block_stats(a[:M])
    bm, bse = _block_stats(b[:M])
    diffs = [a[i] - b[i] for i in range(M)]
    dm, dse = _block_stats(diffs)
    lo, hi = _paired_ci(diffs)
    flag = "CEM STRONGER" if lo > 0 else ("CNN STRONGER" if hi < 0 else "TIE (CI crosses 0)")
    print(f"=== 配对 CEM-vla vs CNN-vla | M={M} identical bench streams ===", flush=True)
    print(f"  CEM-vla : {am:7.1f} +/- {ase:5.1f}", flush=True)
    print(f"  CNN-vla : {bm:7.1f} +/- {bse:5.1f}", flush=True)
    print(f"  paired d(CEM-CNN) = {dm:+7.1f} +/- {dse:5.1f}  95%CI=[{lo:+.1f},{hi:+.1f}]  {flag}",
          flush=True)


def play(seed=0, B=12, wfile="models/cem_w.json"):
    data = json.load(open(wfile, encoding="utf-8"))
    w = data.get("best_w") or data["mu"]
    stream = gen_streams("bench", seed + 1, 50)[seed]
    total, surv = play_cem(stream, w, B)
    print(f"seed={seed} B={B}: total={total:.0f}  surv={surv}/50", flush=True)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "smoke"
    if cmd == "smoke":
        b = 0
        for m in PLACE[6][:1]:
            b, _, _ = apply_mask(b, m)
        print("features(empty):", [round(x, 3) for x in features(0)], flush=True)
        print("features(b):    ", [round(x, 3) for x in features(b)], flush=True)
        w = [-1, -1, -2, -0.5, -0.5, 0.5, 0.5, 0.2]
        st = gen_streams("bench", 1, 50)[0]
        print("greedy w-play total:", play_cem(st, w, 0)[0], flush=True)
        print("beam   w-play total:", play_cem(st, w, 12)[0], flush=True)
    elif cmd == "train":
        a = sys.argv
        cem_train(gens=int(a[2]) if len(a) > 2 else 25,
                  pop=int(a[3]) if len(a) > 3 else 64,
                  elite=int(a[4]) if len(a) > 4 else 12,
                  ngames=int(a[5]) if len(a) > 5 else 24,
                  B=int(a[6]) if len(a) > 6 else 0)
    elif cmd == "bench":
        a = sys.argv
        bench(M=int(a[2]) if len(a) > 2 else 200,
              T=int(a[3]) if len(a) > 3 else 50,
              B=int(a[4]) if len(a) > 4 else 12)
    elif cmd == "look":
        # python cem.py look M T [D S B base cand [Br] [use_leaf]]
        a = sys.argv
        M = int(a[2]) if len(a) > 2 else 100
        T = int(a[3]) if len(a) > 3 else 50
        if len(a) > 8:
            D, S, B = int(a[4]), int(a[5]), int(a[6])
            base, cand = a[7], a[8]
            Br = int(a[9]) if len(a) > 9 else 3
            use_leaf = (a[10].lower() in ("1", "true", "t", "yes", "y")) if len(a) > 10 else True
            bench_look(M=M, T=T, grid=[(D, S, B, base, cand, Br, use_leaf)])
        else:
            bench_look(M=M, T=T)
    elif cmd == "paircfg":
        # python cem.py paircfg base [D S B Br use_leaf] [M T]
        a = sys.argv
        base = a[2] if len(a) > 2 else "beam"
        D = int(a[3]) if len(a) > 3 else 2
        S = int(a[4]) if len(a) > 4 else 6
        B = int(a[5]) if len(a) > 5 else 12
        Br = int(a[6]) if len(a) > 6 else 3
        use_leaf = (a[7].lower() in ("1", "true", "t", "yes", "y")) if len(a) > 7 else True
        M = int(a[8]) if len(a) > 8 else 50
        T = int(a[9]) if len(a) > 9 else 50
        paircfg(D=D, S=S, B=B, base=base, Br=Br, use_leaf=use_leaf, M=M, T=T)
    elif cmd == "perseed":
        perseed()
    elif cmd == "pair":
        pair()
    elif cmd == "play":
        a = sys.argv
        play(seed=int(a[2]) if len(a) > 2 else 0,
             B=int(a[3]) if len(a) > 3 else 12)
    else:
        print("usage: smoke | train | bench | play", flush=True)
