"""
执行：
 1) 健全性检查：bitboard greedy/strong 数值与预期同量级。
 2) 扩展技能阶梯顶端：random↔strong(ε-greedy) + strong + lookahead(D3) + lookahead(D5)。
 3) 收敛判据：strong → LA_D3 → LA_D5，看天花板μ与运气占比的移动是否递减。
 4) 离线上界(hindsight)：rollout 用真实未来牌而非随机，给每个 seed 一个乐观上界。
 5) 交叉点：缺口(到天花板)=σ 处的分数，相对"当前最强玩家"报告(非循环).
"""

import random
import sys
from fast import (NUM_TYPES, greedy_hand, strong_hand, make_lookahead,
                  heuristic, beam_hand, _rollout)


def stdev(xs):
    m = sum(xs) / len(xs)
    return (sum((x - m) ** 2 for x in xs) / len(xs)) ** 0.5


def pct(xs, p):
    s = sorted(xs)
    k = (len(s) - 1) * p / 100
    lo = int(k); hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


# ---- 单局：通用，policy 可带 (seed,move) 也可不带 ----
def play(policy, seed, eps=0.0, needs_ctx=False, max_rounds=100000):
    deal = random.Random(f"deal-{seed}")
    act = random.Random(f"act-{seed}")
    board = 0; combo = 0; total = 0; move = 0
    for _ in range(max_rounds):
        hand = [deal.randrange(NUM_TYPES) for _ in range(3)]
        if eps and act.random() < eps:
            pts, board, combo, alive = greedy_random(board, combo, hand, act)
        elif needs_ctx:
            pts, board, combo, alive = policy(board, combo, hand, act, seed=seed, move=move)
        else:
            pts, board, combo, alive = policy(board, combo, hand, act)
        total += pts; move += 1
        if not alive:
            break
    return total


def greedy_random(board, combo, hand, rng):
    """ε 分支用的纯随机放置。"""
    from fast import PLACE, NCELLS
    from scoring import score_placement, Scoring
    sc = Scoring(); total = 0
    for pid in hand:
        legal = [m for m in PLACE[pid] if not (board & m)]
        if not legal:
            return total, board, combo, False
        from fast import apply_mask
        nb, cl, empty = apply_mask(board, rng.choice(legal))
        pts, combo = score_placement(sc, NCELLS[pid], cl, empty, combo)
        board = nb; total += pts
    return total, board, combo, True


def play_eps(seed, eps, max_rounds=100000):
    """random↔strong 的 ε-greedy 混合(低/中技能轴)。"""
    deal = random.Random(f"deal-{seed}")
    act = random.Random(f"act-{seed}")
    board = 0; combo = 0; total = 0
    for _ in range(max_rounds):
        hand = [deal.randrange(NUM_TYPES) for _ in range(3)]
        if act.random() < eps:
            pts, board, combo, alive = greedy_random(board, combo, hand, act)
        else:
            pts, board, combo, alive = strong_hand(board, combo, hand)
        total += pts
        if not alive:
            break
    return total


# ---- 离线上界：hindsight rollout 用真实未来牌 ----
def play_oracle(seed, D=3, B=12, max_rounds=100000):
    deal = random.Random(f"deal-{seed}")
    act = random.Random(f"act-{seed}")
    board = 0; combo = 0; total = 0
    # 预生成足够长的真实牌流
    stream = [deal.randrange(NUM_TYPES) for _ in range(3 * (max_rounds))]
    pos = 0
    for _ in range(max_rounds):
        hand = stream[pos:pos + 3]
        cands = beam_hand(board, combo, hand, B)
        if not cands:
            break
        # 用"真实未来牌"做 rollout(无随机)，选未来最优
        future = stream[pos + 3: pos + 3 + 3 * D]
        best = None
        for (nb, nc, hs) in cands:
            roll = _rollout_fixed(nb, nc, future)
            v = hs + roll
            if best is None or v > best[0]:
                best = (v, hs, nb, nc)
        total += best[1]; board = best[2]; combo = best[3]
        pos += 3
    return total


def _rollout_fixed(board, combo, future_pieces):
    from fast import apply_mask, PLACE, NCELLS
    from scoring import score_placement, Scoring
    sc = Scoring(); total = 0
    for i in range(0, len(future_pieces), 3):
        hand = future_pieces[i:i + 3]
        for pid in hand:
            best = None
            for m in PLACE[pid]:
                if board & m:
                    continue
                nb, cl, empty = apply_mask(board, m)
                pts, ncb = score_placement(sc, NCELLS[pid], cl, empty, combo)
                val = pts + heuristic(nb)
                if best is None or val > best[0]:
                    best = (val, nb, ncb, pts)
            if best is None:
                return total
            _, board, combo, pts = best
            total += pts
    return total


def summarize(name, scores):
    return (name, sum(scores) / len(scores), stdev(scores),
            pct(scores, 10), pct(scores, 50), pct(scores, 90),
            min(scores), max(scores))


if __name__ == "__main__":
    M = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    ML = int(sys.argv[2]) if len(sys.argv) > 2 else 60   # lookahead 用更少 seed
    print(f"M(ε轴)={M}  M(lookahead/oracle)={ML}\n")

    rows = []
    # 低/中技能轴：ε-greedy random↔strong
    for eps in [1.0, 0.8, 0.6, 0.4, 0.2, 0.0]:
        sc = [play_eps(s, eps) for s in range(M)]
        rows.append(summarize(f"eps={eps:.1f}(1-ε={1-eps:.1f})", sc))
        print("done", rows[-1][0], f"mean={rows[-1][1]:.0f}")

    # 顶端递进：strong 已是 eps=0；加 LA_D3, LA_D5
    la3 = make_lookahead(D=3, S=10, B=12)
    la5 = make_lookahead(D=5, S=10, B=12)
    sc3 = [play(la3, s, needs_ctx=True) for s in range(ML)]
    rows.append(summarize("lookahead_D3", sc3)); print("done LA_D3", f"mean={rows[-1][1]:.0f}")
    sc5 = [play(la5, s, needs_ctx=True) for s in range(ML)]
    rows.append(summarize("lookahead_D5", sc5)); print("done LA_D5", f"mean={rows[-1][1]:.0f}")

    # 离线上界
    orc = [play_oracle(s, D=3) for s in range(ML)]
    rows.append(summarize("ORACLE(hindsight)", orc)); print("done ORACLE", f"mean={rows[-1][1]:.0f}")

    ceiling = max(r[1] for r in rows if not r[0].startswith("ORACLE"))
    print(f"\n天花板(最强非oracle玩家均分) = {ceiling:.0f}")
    print(f"\n{'玩家':>22} {'μ':>8} {'σ':>8} {'p10':>7} {'p50':>7} {'p90':>8} "
          f"{'缺口':>7} {'缺口/σ':>7} {'主导':>5}")
    for (name, m, sd, p10, p50, p90, mn, mx) in rows:
        gap = ceiling - m
        ratio = gap / sd if sd > 0 else 0
        who = "技能" if gap > sd else "运气"
        tag = "(上界)" if name.startswith("ORACLE") else ""
        print(f"{name:>22} {m:8.0f} {sd:8.0f} {p10:7.0f} {p50:7.0f} {p90:8.0f} "
              f"{gap:7.0f} {ratio:7.2f} {who:>5} {tag}")

    # 收敛：strong vs LA_D3 vs LA_D5
    def getrow(n): return next(r for r in rows if r[0] == n or r[0].startswith(n))
    strong_row = getrow("eps=0.0")
    print("\n=== 收敛检查 (顶端递进玩家) ===")
    for nm, r in [("strong", strong_row), ("LA_D3", getrow("lookahead_D3")),
                  ("LA_D5", getrow("lookahead_D5"))]:
        print(f"  {nm:8s} μ={r[1]:7.0f}  σ={r[2]:7.0f}  CV={r[2]/r[1]*100:4.0f}%")
    d_s_3 = getrow("lookahead_D3")[1] - strong_row[1]
    d_3_5 = getrow("lookahead_D5")[1] - getrow("lookahead_D3")[1]
    print(f"  Δ(strong→D3)={d_s_3:.0f}   Δ(D3→D5)={d_3_5:.0f}   "
          f"移动比={d_3_5/d_s_3*100:.0f}% (越小越收敛)")

    # 交叉点
    rs = sorted(rows, key=lambda r: r[1])
    cross = None
    for i in range(len(rs) - 1):
        if rs[i][0].startswith("ORACLE") or rs[i+1][0].startswith("ORACLE"):
            continue
        d1 = ceiling - rs[i][1] - rs[i][2]
        d2 = ceiling - rs[i+1][1] - rs[i+1][2]
        if d1 > 0 >= d2:
            t = d1 / (d1 - d2)
            cross = rs[i][1] + t * (rs[i+1][1] - rs[i][1])
            break
    print("\n" + "=" * 50)
    if cross:
        print(f"交叉点(缺口≈σ) ≈ {cross:.0f}  (相对当前最强玩家，非真最优)")
    else:
        print("未找到交叉点")
