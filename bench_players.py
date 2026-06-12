"""玩家擂台（torch-free，仅依赖 fast.py）：在同一批 CRN 发牌流上比各搜索型玩家的实战得分，
找"得分最高"的可玩策略。目标线①（最强策略），不靠预知未来。
players：strong(beam, 无前瞻) / lookahead(beam + 未来牌蒙特卡洛前瞻, base=greedy|strong)。
用法：python bench_players.py [M=300] [T=50]
"""
import json
import random
import sys
import time

import fast

NT = fast.NUM_TYPES


def gen_streams(tag, M, T):
    """每 seed 一条固定发牌流（T 手 ×3 块）→ 所有玩家玩同一流 = CRN 配对。"""
    out = []
    for m in range(M):
        rng = random.Random(f"{tag}-{m}")
        out.append([[rng.randrange(NT) for _ in range(3)] for _ in range(T)])
    return out


def play_strong(stream, B=12):
    board, combo, total = 0, 0, 0.0
    for rnd, hand in enumerate(stream):
        sc, board, combo, alive = fast.strong_hand(board, combo, hand, B=B)
        if not alive:
            return total, rnd
        total += sc
    return total, len(stream)


def play_policy(stream, policy, seed_idx):
    """lookahead 等 make_* 策略：policy(board,combo,hand,rng,seed,move)。
    rollout RNG 由 (seed,move) 决定，独立于发牌流 → CRN 不破。"""
    board, combo, total = 0, 0, 0.0
    for move, hand in enumerate(stream):
        sc, board, combo, alive = policy(board, combo, hand, None, seed=seed_idx, move=move)
        if not alive:
            return total, move
        total += sc
    return total, len(stream)


def _block_stats(xs, nb=16):
    """mean + 16-block-SE。"""
    M = len(xs)
    blk = M // nb
    bm = [sum(xs[i * blk:(i + 1) * blk]) / blk for i in range(nb)]
    mean = sum(bm) / nb
    se = (sum((x - mean) ** 2 for x in bm) / (nb - 1) / nb) ** 0.5
    return mean, se


def _paired_ci(diffs, n_boot=2000, seed="bench-boot"):
    """配对差 bootstrap 95% CI（禁 ratio-of-means，纯差值）。"""
    M = len(diffs)
    rng = random.Random(seed)
    boots = []
    for _ in range(n_boot):
        s = 0.0
        for _ in range(M):
            s += diffs[rng.randrange(M)]
        boots.append(s / M)
    boots.sort()
    return boots[int(0.025 * n_boot)], boots[int(0.975 * n_boot)]


def bench(M=300, T=50):
    streams = gen_streams("bench", M, T)
    print(f"=== 玩家擂台 | M={M} CRN streams | T={T} | 不靠预知未来 ===", flush=True)

    # 候选玩家（label, 玩一条流的函数 stream->（total,surv））
    LA = fast.make_lookahead
    def LAfn(D, S, B):
        pol = LA(D=D, S=S, B=B, base="greedy")
        return lambda s, i: play_policy(s, pol, i)
    players = [
        ("strong B12", lambda s, i: play_strong(s, B=12)),
        ("look D2 S10 B12", LAfn(2, 10, 12)),
        ("look D3 S10 B12", LAfn(3, 10, 12)),
        ("look D3 S20 B12", LAfn(3, 20, 12)),
        ("look D4 S10 B12", LAfn(4, 10, 12)),
        ("look D3 S10 B20", LAfn(3, 10, 20)),
    ]

    results = {}
    raw = {}
    for label, fn in players:
        t0 = time.time()
        rows = [fn(streams[i], i) for i in range(M)]
        totals = [r[0] for r in rows]
        survs = [r[1] for r in rows]
        mean, se = _block_stats(totals)
        smean, _ = _block_stats([float(x) for x in survs])
        raw[label] = totals
        results[label] = {"mean": mean, "block_se": se, "survival": smean, "wall_s": time.time() - t0}
        print(f"[{label:24s}] mean={mean:7.1f} ± {se:5.1f}  surv={smean:5.1f}/{T}  ({results[label]['wall_s']:.0f}s)", flush=True)

    # 相对 strong B12 的配对差（找谁最强）
    base = raw["strong B12"]
    print(f"\n=== 相对 strong B12 的 CRN 配对差（>0 = 更强）===", flush=True)
    for label in raw:
        if label == "strong B12":
            continue
        diffs = [raw[label][i] - base[i] for i in range(M)]
        dmean, dse = _block_stats(diffs)
        lo, hi = _paired_ci(diffs)
        results[label]["vs_strongB12"] = {"mean": dmean, "block_se": dse, "ci95": [lo, hi]}
        flag = "✓更强" if lo > 0 else ("✗更弱" if hi < 0 else "≈持平")
        print(f"  {label:24s} Δ={dmean:+7.1f} ± {dse:5.1f}  95%CI=[{lo:+.1f},{hi:+.1f}]  {flag}", flush=True)

    best = max(results, key=lambda k: results[k]["mean"])
    print(f"\n-> 最高分玩家 = {best}  (mean={results[best]['mean']:.1f})", flush=True)
    json.dump(results, open("bench_players.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print("-> wrote bench_players.json", flush=True)
    return results


if __name__ == "__main__":
    M = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    T = int(sys.argv[2]) if len(sys.argv) > 2 else 50
    bench(M, T)
