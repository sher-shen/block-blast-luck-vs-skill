"""Step C —— 计分校准（NEXT_STEP_endless_survival.md §3）。

把 scoring.py 参数化为"假设档"(历史默认) vs "真实近似档"(社区逆向公式)，同策略(最强 D2 S30
base=heur)在 T=500 下各跑一套，给"换算到真实计分后大约多少分"。

诚实纪律：
 - 真实 App 公式不公开 → real_approx 是社区逆向、各源不一致 → 仅近似/对外可读，不回头改既有结论。
 - CEM 特征是纯几何(filled/frag/holes/...)、与计分无关 → 权重 w 是棋形评估器、换档仍有效；
   故"换档后让同一策略重优化"是干净的(叶评估不失配)，非把假设档轨迹强行重算。
 - 同 streams(tag=bench) → 发牌 CRN 一致。换计分只改 pts/hand_score/rollout 回报，不改几何。

用法：.venv\\Scripts\\python calibrate_scoring.py [M] [T]
"""
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import scoring
import fast
import cem
from deal_audit import gen_streams_dist, make_cdf, weights_for


def _init_worker(prof):
    """每个 worker 进程启动时设档（Windows=spawn，子进程重新 import 默认 SCORING，
    必须在子进程里设，父进程改了没用 → 这是上一版 Step C 的 bug 根因）。"""
    fast.SCORING = prof
    cem.SCORING = prof


def run(prof_name, prof, deal, M, T, w):
    fast.SCORING = prof   # 父进程也设（gen_streams 不依赖, 但保持一致）
    cem.SCORING = prof
    cdf = make_cdf(weights_for(deal))
    streams = gen_streams_dist("bench", M, T, cdf)
    D, S, B, base, cand, Br, use_leaf = 2, 30, 12, "heur", "cem", 3, True
    argz = [(streams[i], w, D, S, B, base, cand, i, Br, use_leaf) for i in range(M)]
    with ProcessPoolExecutor(initializer=_init_worker, initargs=(prof,)) as pool:
        rows = list(pool.map(cem._look_worker, argz))
    totals = [r[0] for r in rows]
    survs = [float(r[1]) for r in rows]
    tmean, tse = cem._block_stats(totals)
    smean, sse = cem._block_stats(survs)
    return {"profile": prof_name, "deal": deal, "total": tmean, "total_se": tse,
            "surv": smean, "surv_se": sse,
            "censored": sum(1 for x in survs if x >= T),
            "totals": totals, "survs": [int(x) for x in survs]}


def main():
    M = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    T = int(sys.argv[2]) if len(sys.argv) > 2 else 500
    data = json.load(open("models/cem_w.json", encoding="utf-8"))
    w = data.get("best_w") or data["mu"]
    cfgs = [
        ("assumed",     scoring.assumed(),     "uniform"),   # = Step A 基线 ~5954
        ("real_approx", scoring.real_approx(), "uniform"),   # 隔离计分映射效应
        ("real_approx", scoring.real_approx(), "no_hard"),   # A+B+C 合并: 贴近真实条件估计
    ]
    print(f"=== Step C 计分校准 | M={M} T={T} | D2 S30 base=heur | 同 streams CRN ===", flush=True)
    res = []
    for name, prof, deal in cfgs:
        t0 = time.time()
        r = run(name, prof, deal, M, T, w)
        res.append(r)
        print(f"[{name:11s} | deal={deal:8s}] total={r['total']:8.0f}+/-{r['total_se']:5.0f}"
              f"  surv={r['surv']:6.1f}+/-{r['surv_se']:4.1f}  censored@T={r['censored']}/{M}"
              f"  ({time.time()-t0:.0f}s)", flush=True)
    # 计分映射倍率（同 uniform 发牌, real vs assumed）
    a = next(r for r in res if r["profile"] == "assumed")
    ru = next(r for r in res if r["profile"] == "real_approx" and r["deal"] == "uniform")
    print(f"\n计分映射倍率 (uniform 发牌, real/assumed): total {ru['total']/a['total']:.2f}× "
          f"(surv {ru['surv']/a['surv']:.2f}× ⇒ 主要是计分映射, 非生存)", flush=True)
    json.dump(res, open("calibrate_scoring.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print("-> wrote calibrate_scoring.json", flush=True)


if __name__ == "__main__":
    main()
