"""
oracle_realistic.py — Direction A' (续20)：把 luck/skill (oracle 双通道 EVPI) 重测在
**更真实的 benchmark 设定**下，回答续17/18 留的问题"占比 57-69% 对 benchmark 定义稳不稳"。

== 为什么是 A' 而不是朴素 A（独立审核的 BLOCKER）==
朴素"搬到无限局 T=500"会让 EVPI 估计量依赖的【全员存活 cohort】坍塌（strong 几乎全死、
seer 近乎不死、无界 horizon 分数爆炸）——精确重现 oracle_immortality_reframe.md 撤回过的
"78% 口径坏掉"病理。固定 horizon T 当初正是为此引入。故 A' **守住固定中等 T**，只变两个
**良定义**的 benchmark 旋钮：
  axis-1 计分:  assumed（历史假设档） vs real_approx（社区逆向乘法连击档，scoring.real_approx）
  axis-2 发牌:  uniform（均匀 38 型，原设定） vs benevolent（去 3x3 = deal_audit 的 no9，
                续18 实测最温和却 ×2.7 生存的善意档）
2×2 析因，固定 Ts=[40,50,60]、D_oracle=3、CRN 配对，复用 oracle_analysis 的玩家与统计口径。

== 良定义 / 防坑（审核要求）==
  * 固定 T ⇒ 分数有限、cohort 有定义；善意发牌只会**增大** cohort（更多人活到 T），更稳。
  * (assumed, uniform) 这格用 oracle_analysis **原函数**跑 ⇒ 必复现 channelB 的 69/65/57%
    = 自洽 sanity（同 599 一样的复现检验）。
  * SCORING 在 **worker 内每任务设置**（fast.SCORING），躲开续18 的 Windows-spawn
    ProcessPool 默认 SCORING 坑（worker 重 import → 不继承父进程改动）。
  * 善意发牌**穿透到 blind/seer 的内部 rollout 采样器**（用同一 benevolent CDF 抽虚构未来），
    否则 blind 的边缘化器相对真实发牌分布失配、EVPI 含义被改（审核 risk #2）。
  * 主口径 = **strong-分母三玩家 EVPI 占比**：strong/blind/seer 都直接读 fast.SCORING 且消费
    发牌流 ⇒ 任何 scoring/dealing 下都**自动重新最优化**，无 OOD 混淆。
    vla 的训练 V 锁死在 assumed+uniform，real_approx 下是 OOD（叶值在错币种）⇒ A' **不**把 vla
    当分母（避免污染良定义结论）；vla-真实化是单独的 OOD 部署问题，诚实留作后续。
  * 写**新文件** evpi_realistic.json + 新模块 ⇒ oracle_analysis.py / channelB.json / scoring.py
    全部 byte-unchanged，n=25 与既有 headline firewall 不破。

零依赖（torch-free），复用 fast.py 原语 + oracle_analysis 统计。
"""
import json
import random
import sys
import time
from bisect import bisect_right
from concurrent.futures import ProcessPoolExecutor

import fast
import scoring as scoring_mod
import oracle_analysis as oa
from fast import NUM_TYPES, beam_hand, heuristic, _rollout_fixed_strong

B_ROLL = 6                       # 与 oracle_analysis 一致
D_ORACLE = 3
TS = (40, 50, 60)
SIZES = [len(c) for c in fast.PIECE_CELLS]

SCORINGS = {"assumed": scoring_mod.assumed, "real_approx": scoring_mod.real_approx}
DEALINGS = ("uniform", "benevolent")     # benevolent = no9（去 3x3）


# ---------- 发牌分布（镜像 deal_audit，benevolent=no9） ----------
def weights_for(dealing):
    if dealing == "uniform":
        return [1.0] * NUM_TYPES
    if dealing == "benevolent":          # = deal_audit 的 "no9"：去掉 3x3 九格
        return [0.0 if sz == 9 else 1.0 for sz in SIZES]
    raise ValueError(dealing)


def make_cdf(w):
    tot = sum(w)
    acc, cdf = 0.0, []
    for x in w:
        acc += x / tot
        cdf.append(acc)
    cdf[-1] = 1.0
    return cdf


# ---------- benevolent 发牌下的玩家变体（镜像 oracle_analysis，唯一差别=抽牌走 inverse-CDF） ----------
# uniform 发牌时**不**用这些；直接调 oa.play_*（保证与 channelB 字节级一致）。
def _bdeal_stream(seed, T, cdf):
    deal = random.Random(f"deal-{seed}")
    return [bisect_right(cdf, deal.random()) for _ in range(3 * (T + 16))]


def _play_strong_b(seed, T, cdf, B=12):
    stream = _bdeal_stream(seed, T, cdf)
    board = 0; combo = 0; total = 0; rounds = 0; pos = 0
    for _ in range(T):
        from fast import strong_hand
        hs, nb, nc, alive = strong_hand(board, combo, stream[pos:pos + 3], B=B)
        if not alive:
            break
        total += hs; board = nb; combo = nc; rounds += 1; pos += 3
    return total, rounds


def _play_lookahead_b(seed, future_src, cdf, D=5, B=12, S=8, T=100, use_heur=True):
    """镜像 oracle_analysis.play_lookahead，但真实牌流与内部虚构未来都从 benevolent cdf 抽。"""
    stream = _bdeal_stream(seed, T, cdf)
    board = 0; combo = 0; total = 0; rounds = 0; pos = 0
    for move in range(T):
        hand = stream[pos:pos + 3]
        if future_src == "real":                      # seer：真实(=benevolent)未来
            fut = stream[pos + 3: pos + 3 + 3 * D]
            voc = lambda nb, nc, _f=fut: _rollout_fixed_strong(nb, nc, _f, B=B_ROLL)
        else:                                          # sampled_avg：S 份 benevolent 采样未来取平均
            futs = [[bisect_right(cdf, random.Random(f"blind-{seed}-{move}-{s}").random())
                     for _ in range(3 * D)] for s in range(S)]
            voc = lambda nb, nc, _f=futs: sum(
                _rollout_fixed_strong(nb, nc, f, B=B_ROLL) for f in _f) / len(_f)
        hs, nb, nc, alive = oa._lookahead_step(board, combo, hand, B, voc, use_heur)
        if not alive:
            break
        total += hs; board = nb; combo = nc; rounds += 1; pos += 3
    return total, rounds


# ---------- worker：每任务设 SCORING（躲 spawn 坑） + 选 uniform/benevolent 路径 ----------
def _worker(arg):
    seed, T, scoring_tag, dealing = arg
    # 关键：在 worker 进程内设 SCORING（fast 与 oracle_analysis 都按名读 fast.SCORING）
    fast.SCORING = SCORINGS[scoring_tag]()
    if dealing == "uniform":
        st = oa.play_strong(seed, T=T)
        bl = oa.play_blind(seed, D=D_ORACLE, T=T)
        se = oa.play_seer(seed, D=D_ORACLE, T=T)
    else:
        cdf = make_cdf(weights_for(dealing))
        st = _play_strong_b(seed, T, cdf)
        bl = _play_lookahead_b(seed, "sampled_avg", cdf, D=D_ORACLE, T=T)
        se = _play_lookahead_b(seed, "real", cdf, D=D_ORACLE, T=T)
    return (seed, T, scoring_tag, dealing, st[0], st[1], bl[0], bl[1], se[0], se[1])


# ---------- 分析（口径与 oracle_analysis.channel_analysis 完全一致） ----------
def _share(seer, blind, strong, cohort):
    vals = [(seer[i] - blind[i]) / (seer[i] - strong[i])
            for i in cohort if seer[i] - strong[i] > 0]
    if not vals:
        return {"median": float("nan"), "ci": [float("nan")] * 2, "n_valid": 0,
                "n_cohort": len(cohort)}
    m, lo, hi = oa.bootstrap_ci(vals, oa._median, boot_seed="evpi-real-share")
    return {"median": m, "ci": [lo, hi], "n_valid": len(vals), "n_cohort": len(cohort)}


def analyze(rows, seeds):
    # rows: list of worker outputs. index by (scoring,dealing,T,seed).
    by = {}
    for (seed, T, sc, dl, ss, sr, bs, br, es, er) in rows:
        by.setdefault((sc, dl, T), {})[seed] = {
            "strong": (ss, sr), "blind": (bs, br), "seer": (es, er)}
    out = {"N": len(seeds), "D_oracle": D_ORACLE, "Ts": list(TS),
           "axes": {"scoring": list(SCORINGS), "dealing": list(DEALINGS)},
           "benevolent_def": "no9 (drop 3x3, mirrors deal_audit)", "conditions": {}}
    for sc in SCORINGS:
        for dl in DEALINGS:
            cond = f"{sc}_{dl}"
            out["conditions"][cond] = {}
            print(f"\n########## condition: scoring={sc}  dealing={dl} ##########", flush=True)
            for T in TS:
                d = by[(sc, dl, T)]
                S = {k: [d[s][k][0] for s in seeds] for k in ("strong", "blind", "seer")}
                R = {k: [d[s][k][1] for s in seeds] for k in ("strong", "blind", "seer")}
                surv = {k: sum(1 for r in R[k] if r >= T) / len(seeds) for k in S}
                cohort = [i for i in range(len(seeds))
                          if all(R[k][i] >= T for k in ("strong", "blind", "seer"))]
                share = _share(S["seer"], S["blind"], S["strong"], cohort)
                # 带符号分解（均值, cohort）
                evpi = [S["seer"][i] - S["blind"][i] for i in cohort]
                proc = [S["blind"][i] - S["strong"][i] for i in cohort]
                raw = [S["seer"][i] - S["strong"][i] for i in cohort]
                rec = {
                    "surv": surv,
                    "score_median_full": {k: oa.pct(S[k], 50) for k in S},
                    "cohort_n": len(cohort),
                    "score_median_cohort": {
                        k: (oa.pct([S[k][i] for i in cohort], 50) if cohort else float("nan"))
                        for k in S},
                    "EVPI_share_strong_denom": share,
                    "decomp_mean": {
                        "raw_seer_strong": (sum(raw) / len(raw)) if raw else float("nan"),
                        "EVPI_seer_blind": (sum(evpi) / len(evpi)) if evpi else float("nan"),
                        "procedure_blind_strong": (sum(proc) / len(proc)) if proc else float("nan"),
                    },
                }
                out["conditions"][cond][f"T{T}"] = rec
                sh = share["median"]
                print(f"  T={T:>2}  cohort={len(cohort):>3}/{len(seeds)}  "
                      f"surv[st/bl/se]={surv['strong']:.2f}/{surv['blind']:.2f}/{surv['seer']:.2f}  "
                      f"score_med[st/bl/se]={rec['score_median_cohort']['strong']:.0f}/"
                      f"{rec['score_median_cohort']['blind']:.0f}/"
                      f"{rec['score_median_cohort']['seer']:.0f}  "
                      f"EVPI-share={sh*100:.0f}% "
                      f"[{share['ci'][0]*100:.0f},{share['ci'][1]*100:.0f}] "
                      f"n={share['n_valid']}/{share['n_cohort']}", flush=True)
    return out


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 120
    seeds = list(range(N))
    tasks = [(s, T, sc, dl) for sc in SCORINGS for dl in DEALINGS
             for T in TS for s in seeds]
    print(f"=== A' EVPI realistic sensitivity | N={N} D_oracle={D_ORACLE} Ts={list(TS)} "
          f"| 2x2 scoring x dealing = {len(tasks)} tasks ===", flush=True)
    t0 = time.time()
    rows = []
    with ProcessPoolExecutor() as pool:
        for i, r in enumerate(pool.map(_worker, tasks, chunksize=4)):
            rows.append(r)
            if (i + 1) % 120 == 0:
                print(f"  ... {i+1}/{len(tasks)} games ({time.time()-t0:.0f}s)", flush=True)
    out = analyze(rows, seeds)
    json.dump(out, open("evpi_realistic.json", "w"), indent=2)
    print(f"\n-> wrote evpi_realistic.json  (total {time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
