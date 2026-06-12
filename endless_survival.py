"""Step A —— 无限局 + 生存分布审计（NEXT_STEP_endless_survival.md §1）。

把 T 从 50 拉到 500（远超 surv≈42，封顶不再 binding ≈ 无限局），跑最强配置
(D2 S30 B12 base=heur cand=cem)，看生存回合分布 + 总分分布。

play_cem_look 死亡时提前返回 (total, move=死亡回合) → 无需改逻辑，只传长流。
rollout 未来流 RNG 由 (seed_idx, move) 定、与 T 无关 → CRN 不破。
streams 用 tag="bench"、与历史 T=50 bench 同种子 → 前 50 手逐字相同 → 生存数可直接对比历史。

用法：.venv\\Scripts\\python endless_survival.py [M] [T]
"""
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import cem


def pct(xs, p):
    xs = sorted(xs)
    if not xs:
        return float("nan")
    k = (len(xs) - 1) * p / 100.0
    lo = int(k)
    hi = min(lo + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def histogram(xs, lo, hi, step):
    edges = list(range(lo, hi + step, step))
    counts = [0] * (len(edges) - 1)
    for x in xs:
        for i in range(len(edges) - 1):
            if edges[i] <= x < edges[i + 1]:
                counts[i] += 1
                break
    return edges, counts


def main():
    M = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    T = int(sys.argv[2]) if len(sys.argv) > 2 else 500
    wfile = "models/cem_w.json"
    data = json.load(open(wfile, encoding="utf-8"))
    w = data.get("best_w") or data["mu"]
    D, S, B, base, cand, Br, use_leaf = 2, 30, 12, "heur", "cem", 3, True

    streams = cem.gen_streams("bench", M, T)
    print(f"=== Step A 无限局生存审计 | M={M} bench-CRN | T={T} | "
          f"D{D} S{S} B{B} base={base} cand={cand} | 不预知未来 ===", flush=True)
    t0 = time.time()
    argz = [(streams[i], w, D, S, B, base, cand, i, Br, use_leaf) for i in range(M)]
    with ProcessPoolExecutor() as pool:
        rows = list(pool.map(cem._look_worker, argz))
    dt = time.time() - t0

    totals = [r[0] for r in rows]
    survs = [int(r[1]) for r in rows]
    censored = sum(1 for s in survs if s >= T)  # 撞到 T 封顶（未自然死）

    tmean, tse = cem._block_stats(totals)
    smean, sse = cem._block_stats([float(x) for x in survs])

    print(f"\n--- 生存回合 surv (自然死=活到第几回合; 封顶 T={T}) ---", flush=True)
    print(f"  mean={smean:6.2f} +/-{sse:4.2f}  min={min(survs)}  max={max(survs)}", flush=True)
    print(f"  pctl  p10={pct(survs,10):.0f}  p25={pct(survs,25):.0f}  "
          f"p50={pct(survs,50):.0f}  p75={pct(survs,75):.0f}  p90={pct(survs,90):.0f}", flush=True)
    print(f"  撞 T={T} 封顶(未自然死): {censored}/{M} = {100*censored/M:.0f}%", flush=True)
    edges, counts = histogram(survs, 0, max(60, ((max(survs)//10)+1)*10), 10)
    for i, c in enumerate(counts):
        bar = "#" * c
        print(f"    [{edges[i]:3d},{edges[i+1]:3d})  {c:3d}  {bar}", flush=True)

    print(f"\n--- 总分 total ---", flush=True)
    print(f"  mean={tmean:7.1f} +/-{tse:5.1f}  min={min(totals):.0f}  max={max(totals):.0f}", flush=True)
    print(f"  pctl  p10={pct(totals,10):.0f}  p25={pct(totals,25):.0f}  "
          f"p50={pct(totals,50):.0f}  p75={pct(totals,75):.0f}  p90={pct(totals,90):.0f}", flush=True)
    print(f"\n  ({dt:.0f}s)", flush=True)

    out = {
        "M": M, "T": T, "config": f"D{D} S{S} B{B} base={base} cand={cand}",
        "surv_mean": smean, "surv_se": sse, "surv_min": min(survs), "surv_max": max(survs),
        "surv_pctl": {p: pct(survs, p) for p in (10, 25, 50, 75, 90)},
        "censored_at_T": censored, "censored_frac": censored / M,
        "total_mean": tmean, "total_se": tse,
        "total_pctl": {p: pct(totals, p) for p in (10, 25, 50, 75, 90)},
        "survs": survs, "totals": totals, "secs": dt,
    }
    json.dump(out, open("endless_survival.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print("-> wrote endless_survival.json", flush=True)


if __name__ == "__main__":
    main()
