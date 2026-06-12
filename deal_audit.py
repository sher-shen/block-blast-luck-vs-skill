"""Step B —— 发牌分布审计（NEXT_STEP_endless_survival.md §2）。

假设：均匀随机 38 型(含 3x3/2x3/五连等难块)比真实游戏难得多 → 逼死策略。
做法：构造若干"偏善意发牌"变体(静态重加权 piece 概率)，CRN 配对(同 per-stream 种子的
同一串 uniform 抽样经 inverse-CDF 映到不同分布 → 耦合) 重测最强策略的生存 + 分。

诚实边界：
 - 真实 App 发牌机制不公开 → 这是"若发牌更友善则…"的敏感性分析，不宣称复现真实 App。
 - play_cem_look 内部 rollout 仍按 uniform-38 抽未来 → 这正是"uniform 训练的策略部署到善意真游戏"
   的真实场景(策略不知真分布)；属诚实设定，已标注。
 - 静态重加权 = 板态无关 → 不含"保证至少一块可下/避免连续死手"这类板态相关善意机制(那会耦合策略,
   破 CRN)。故这是善意发牌的**下界**:真实善意机制只会更救命。

用法：.venv\\Scripts\\python deal_audit.py [M] [T]
"""
import json
import sys
import time
import random
from bisect import bisect_right
from concurrent.futures import ProcessPoolExecutor

import cem
import fast

SIZES = [len(c) for c in fast.PIECE_CELLS]  # 每个 piece-id 的格数


def weights_for(variant):
    """返回长度 NUM_TYPES 的权重向量。"""
    w = []
    for sz in SIZES:
        if variant == "uniform":
            w.append(1.0)
        elif variant == "no9":            # 去掉 3x3 九格
            w.append(0.0 if sz == 9 else 1.0)
        elif variant == "no_hard":        # 去掉 size>=6 (3x3 + 2x3)
            w.append(0.0 if sz >= 6 else 1.0)
        elif variant == "no_ge5":         # 去掉所有 size>=5 难块
            w.append(0.0 if sz >= 5 else 1.0)
        elif variant == "small_bias":     # 权重 ∝ 1/size^2，强烈偏小块
            w.append(1.0 / (sz * sz))
        else:
            raise ValueError(variant)
    return w


def make_cdf(w):
    tot = sum(w)
    acc, cdf = 0.0, []
    for x in w:
        acc += x / tot
        cdf.append(acc)
    cdf[-1] = 1.0
    return cdf


def gen_streams_dist(tag, M, T, cdf):
    """CRN 耦合发牌：每 slot 用 per-stream 种子抽 u~U(0,1) -> inverse-CDF 映到 pid。
    同 (tag,m) 在所有 variant 下 u 序列相同 -> 配对耦合。"""
    out = []
    for m in range(M):
        rng = random.Random(f"{tag}-{m}")
        stream = [[bisect_right(cdf, rng.random()) for _ in range(3)] for _ in range(T)]
        out.append(stream)
    return out


def run_variant(variant, M, T, w, pool):
    cdf = make_cdf(weights_for(variant))
    streams = gen_streams_dist("bench", M, T, cdf)
    D, S, B, base, cand, Br, use_leaf = 2, 30, 12, "heur", "cem", 3, True
    argz = [(streams[i], w, D, S, B, base, cand, i, Br, use_leaf) for i in range(M)]
    rows = list(pool.map(cem._look_worker, argz))
    totals = [r[0] for r in rows]
    survs = [float(r[1]) for r in rows]
    tmean, tse = cem._block_stats(totals)
    smean, sse = cem._block_stats(survs)
    return {"variant": variant, "surv": smean, "surv_se": sse,
            "total": tmean, "total_se": tse,
            "survs": [int(x) for x in survs], "totals": totals,
            "censored": sum(1 for x in survs if x >= T)}


def paired_delta(a, b):
    """a-b 的配对差(同 stream 索引) + 16-block-SE CI。"""
    da = [a["totals"][i] - b["totals"][i] for i in range(len(a["totals"]))]
    m, se = cem._block_stats(da)
    ds = [a["survs"][i] - b["survs"][i] for i in range(len(a["survs"]))]
    sm, sse = cem._block_stats([float(x) for x in ds])
    return m, se, sm, sse


def main():
    M = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    T = int(sys.argv[2]) if len(sys.argv) > 2 else 500
    data = json.load(open("models/cem_w.json", encoding="utf-8"))
    w = data.get("best_w") or data["mu"]
    variants = ["uniform", "no9", "no_hard", "no_ge5", "small_bias"]
    print(f"=== Step B 发牌审计 | M={M} T={T} | D2 S30 base=heur | rollout 仍 uniform-38 ===",
          flush=True)
    t0 = time.time()
    res = {}
    with ProcessPoolExecutor() as pool:
        for v in variants:
            tv = time.time()
            res[v] = run_variant(v, M, T, w, pool)
            r = res[v]
            print(f"[{v:11s}] surv={r['surv']:6.1f}+/-{r['surv_se']:4.1f}  "
                  f"total={r['total']:7.0f}+/-{r['total_se']:5.0f}  "
                  f"censored@T={r['censored']}/{M}  ({time.time()-tv:.0f}s)", flush=True)
    print(f"\n--- 配对 delta vs uniform (正=善意发牌更强) ---", flush=True)
    base = res["uniform"]
    for v in variants:
        if v == "uniform":
            continue
        dt, dse, ds, dsse = paired_delta(res[v], base)
        print(f"  {v:11s}  Δtotal={dt:+7.0f}+/-{dse:5.0f}  Δsurv={ds:+6.1f}+/-{dsse:4.1f}",
              flush=True)
    print(f"\n({time.time()-t0:.0f}s)", flush=True)
    json.dump(res, open("deal_audit.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print("-> wrote deal_audit.json", flush=True)


if __name__ == "__main__":
    main()
