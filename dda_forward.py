"""
dda_forward.py — Direction B (续20)：把"运气"当**可设计的旋钮**做正向敏感性扫描
（DDA = dynamic difficulty adjustment 的正向问题）。

== 问题 ==
目标线② 把 luck-share(EVPI 占比) 操作化成可计算量。反过来：发牌分布是设计者唯一能直接
拧的旋钮（计分公式、棋盘大小通常固定）。**给定一族发牌分布，luck-share 怎么随难度变？**
若 luck-share 随发牌难度单调/可预测地变 ⇒ 设计者能据此"设定运气含量"（DDA 的核心诉求）。

== 为什么只做正向（独立审核的 de-scope）==
完整反问题"给定目标 luck-share 反解整个发牌分布"是一个独立的中型优化项目（高维、非凸、
每次评估要跑一遍 oracle）。审核建议：先做**正向 1-D 扫描 + 在曲线上读出反解**（要 luck-share=X
就插值出 β=…），便宜、诚实、信息量足，不捆绑优化器。

== 设计 ==
单参数发牌族：w_i ∝ exp(-β · size_i)（size_i=piece i 的格数）。
  β = 0   → 均匀（原设定）
  β > 0   → 偏好小块 = 善意/简单（生存↑）
  β < 0   → 偏好大块 = 对抗/困难（生存↓）
扫 β，对每个 β 跑 strong/blind/seer（固定 T、CRN、复用 oracle_realistic 的玩家与口径），
报 (生存率, EVPI 占比, 分数阶梯)。输出 dda_forward.json + 设计曲线。

诚实边界（沿用 deal_audit）：静态板态无关重加权 = 善意发牌的**下界**（不含"保证可下"等板态相关
善意）；rollout 内部虚构未来也按同一 β 分布抽（边缘化器正确指定，见 oracle_realistic）。
固定中等 T ⇒ EVPI 良定义（同 A' 的 BLOCKER 规避）。计分固定 assumed（headline 币种）。

零依赖；复用 oracle_realistic 的 benevolent 玩家路径（任意发牌 cdf）。
"""
import json
import math
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import fast
import scoring as scoring_mod
import oracle_analysis as oa
import oracle_realistic as orl

D_ORACLE = 3
T_FIX = 50                       # 单 T 扫描（保持便宜）；A' 已覆盖 T 维
SIZES = orl.SIZES
# β 网格：负=对抗(偏大块/难)，0=均匀，正=善意(偏小块/易)
BETAS = (-0.20, -0.10, -0.05, 0.0, 0.05, 0.10, 0.20, 0.30, 0.40)


def cdf_for_beta(beta):
    w = [math.exp(-beta * sz) for sz in SIZES]
    return orl.make_cdf(w)


def mean_dealt_size(cdf):
    """该发牌分布下每块的期望格数（难度的可读标量）。"""
    probs = []
    prev = 0.0
    for c in cdf:
        probs.append(c - prev); prev = c
    return sum(p * sz for p, sz in zip(probs, SIZES))


def _worker(arg):
    seed, T, beta = arg
    fast.SCORING = scoring_mod.assumed()      # headline 币种；worker 内设(躲 spawn 坑)
    cdf = cdf_for_beta(beta)
    st = orl._play_strong_b(seed, T, cdf)
    bl = orl._play_lookahead_b(seed, "sampled_avg", cdf, D=D_ORACLE, T=T)
    se = orl._play_lookahead_b(seed, "real", cdf, D=D_ORACLE, T=T)
    return (seed, beta, st[0], st[1], bl[0], bl[1], se[0], se[1])


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    seeds = list(range(N))
    tasks = [(s, T_FIX, b) for b in BETAS for s in seeds]
    print(f"=== B: forward DDA sweep | N={N} D_oracle={D_ORACLE} T={T_FIX} "
          f"| betas={BETAS} = {len(tasks)} tasks ===", flush=True)
    t0 = time.time()
    by = {b: {} for b in BETAS}
    with ProcessPoolExecutor() as pool:
        for (seed, beta, ss, sr, bs, br, es, er) in pool.map(_worker, tasks, chunksize=4):
            by[beta][seed] = {"strong": (ss, sr), "blind": (bs, br), "seer": (es, er)}
    print(f"[sweep] done in {time.time()-t0:.0f}s", flush=True)

    out = {"N": N, "D_oracle": D_ORACLE, "T": T_FIX, "betas": list(BETAS),
           "family": "w_i ∝ exp(-beta*size_i); beta>0=benevolent(small), beta<0=adversarial(big)",
           "scoring": "assumed", "points": []}
    print(f"\n{'beta':>6} {'E[size]':>7} {'cohort':>7} {'surv_st':>7} {'surv_bl':>7} "
          f"{'sc_st':>6} {'sc_se':>6} {'EVPI%':>6} {'CI':>14}", flush=True)
    for b in BETAS:
        d = by[b]
        S = {k: [d[s][k][0] for s in seeds] for k in ("strong", "blind", "seer")}
        R = {k: [d[s][k][1] for s in seeds] for k in ("strong", "blind", "seer")}
        surv = {k: sum(1 for r in R[k] if r >= T_FIX) / N for k in S}
        cohort = [i for i in range(N) if all(R[k][i] >= T_FIX for k in S)]
        share = orl._share(S["seer"], S["blind"], S["strong"], cohort)
        esz = mean_dealt_size(cdf_for_beta(b))
        sc_st = oa.pct([S["strong"][i] for i in cohort], 50) if cohort else float("nan")
        sc_se = oa.pct([S["seer"][i] for i in cohort], 50) if cohort else float("nan")
        rec = {"beta": b, "mean_dealt_size": esz, "cohort_n": len(cohort),
               "surv": surv, "EVPI_share": share,
               "score_median_cohort": {
                   k: (oa.pct([S[k][i] for i in cohort], 50) if cohort else float("nan"))
                   for k in S}}
        out["points"].append(rec)
        sh = share["median"]
        print(f"{b:>6.2f} {esz:>7.2f} {len(cohort):>7} {surv['strong']:>7.2f} "
              f"{surv['blind']:>7.2f} {sc_st:>6.0f} {sc_se:>6.0f} "
              f"{sh*100:>5.0f}% [{share['ci'][0]*100:>4.0f},{share['ci'][1]*100:>4.0f}] "
              f"n={share['n_valid']}/{share['n_cohort']}", flush=True)
    json.dump(out, open("dda_forward.json", "w"), indent=2)
    print(f"\n-> wrote dda_forward.json  (total {time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
