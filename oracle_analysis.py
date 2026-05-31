"""
oracle_analysis.py — 让"运气=信息价值(oracle 缺口)"这个 headline 数字变严谨。

两轮独立审核后的设计（见 memory/log.md 2026-05-31）：
  1. 主估计量不是 ratio-of-means(被长尾拉高)，而是 **per-seed 配对差** d_i = seer_i - strong_i，
     报 median + IQR + bootstrap 95% CI（对 mean 和 median 都报）。
  2. oracle 不再是 D=3 greedy 偷看，而是 **beam rollout + D-sweep 到 plateau**，
     明确标注为"可实现前瞻的下界(realizable-foresight lower bound)"，不是"真最优"。
  3. 把缺口**可识别地分解**为 信息价值(luck) vs 在线次优残差(skill)：
       strong  = 最强在线玩家(beam_hand，无前瞻)
       blind   = 前瞻玩家，rollout 用"采样的随机未来"(S 份取平均)
       seer    = 同一前瞻玩家，rollout 用"那一条真实未来"
     blind 与 seer 唯一差别 = 信息集(采样未来 vs 真实未来)，基策略/D/B/打分公式全同。
       VoI(luck)      = seer - blind
       residual(skill)= blind - strong
       total gap      = seer - strong = VoI + residual
  4. CRN：三玩家共用 deal-{seed} 牌流(按抽牌序消费)，棋盘发散但配对差方差大减。
     缺口为总分(per-game)量；额外报 rounds-survived 与 per-round 缺口，避免暗示同 horizon。

零依赖，复用 fast.py 原语。
"""

import random
import sys

from fast import (NUM_TYPES, beam_hand, strong_hand, heuristic,
                  _rollout_fixed_strong)

B_ROLL = 6   # rollout 内部 beam 宽度（与 make_lookahead strong-base 一致）


# ---------- 工具 ----------
def stdev(xs):
    m = sum(xs) / len(xs)
    return (sum((x - m) ** 2 for x in xs) / len(xs)) ** 0.5


def pct(xs, p):
    s = sorted(xs)
    if len(s) == 1:
        return s[0]
    k = (len(s) - 1) * p / 100
    lo = int(k); hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def bootstrap_ci(xs, stat=lambda v: sum(v) / len(v), nboot=2000, alpha=0.05,
                 boot_seed="boot"):
    """对一维样本做 seeded bootstrap，返回 (point, lo, hi)。stat 默认 mean。"""
    n = len(xs)
    rng = random.Random(boot_seed)
    reps = []
    for _ in range(nboot):
        sample = [xs[rng.randrange(n)] for _ in range(n)]
        reps.append(stat(sample))
    reps.sort()
    lo = reps[int((alpha / 2) * nboot)]
    hi = reps[int((1 - alpha / 2) * nboot) - 1]
    return stat(xs), lo, hi


def _median(v):
    return pct(v, 50)


# 单侧 95% Poisson 上界率：k 死亡 / exposure 在险轮。mu_up = 0.5*chi2_{0.95}(df=2(k+1))。
# 无 scipy → 查表 χ²_{0.95} 常数（单侧上界用 0.95 分位，非 0.975 双侧）。审核验证 k=1 → 3.30e-4。
_CHI2_095 = {2: 5.9915, 4: 9.4877, 6: 12.5916, 8: 15.5073, 10: 18.3070}


def poisson_upper_95(k, exposure):
    """k 个事件 / exposure 暴露量的单侧 95% Poisson 上界率。返回 (ub_rate, mu_up)。"""
    df = 2 * (k + 1)
    chi = _CHI2_095.get(df)
    if chi is None:  # Wilson–Hilferty 近似兜底
        z = 1.6448536269514722  # Phi^{-1}(0.95)
        chi = df * (1 - 2.0 / (9 * df) + z * (2.0 / (9 * df)) ** 0.5) ** 3
    mu_up = 0.5 * chi
    return (mu_up / exposure if exposure else float("nan")), mu_up


# ---------- 玩家：共用真实牌流，CRN 配对 ----------
# 发现(2026-05-31)：能看真实未来的 seer + beam rollout → 近乎不死，无界horizon下分数
# 爆炸(百万级)。故所有比较改为【固定 horizon T】(同样跑 T 轮，分数才可比、有限)，
# 并把"运气"拆成两条通道：存活运气(活过 T 的概率) + 计分运气(给定存活的信息价值/EVPI)。
def _deal_stream(seed, max_rounds):
    deal = random.Random(f"deal-{seed}")
    return [deal.randrange(NUM_TYPES) for _ in range(3 * (max_rounds + 16))]


def play_strong(seed, T=100, B=12):
    """最强在线玩家：每手 beam-strong，无前瞻。返回 (score, rounds_survived)。"""
    stream = _deal_stream(seed, T)
    board = 0; combo = 0; total = 0; rounds = 0; pos = 0
    for _ in range(T):
        hs, nb, nc, alive = strong_hand(board, combo, stream[pos:pos + 3], B=B)
        if not alive:
            break
        total += hs; board = nb; combo = nc; rounds += 1; pos += 3
    return total, rounds


def _lookahead_step(board, combo, hand, B, value_of_cand, use_heur=True):
    """beam 出候选 → value_of_cand(nb,nc) 估未来 → 选 hand_score(+heur)+未来 最大者。"""
    cands = beam_hand(board, combo, hand, B)
    if not cands:
        return 0, board, combo, False
    best = None
    for (nb, nc, hs) in cands:
        v = hs + value_of_cand(nb, nc) + (heuristic(nb) if use_heur else 0.0)
        if best is None or v > best[0]:
            best = (v, hs, nb, nc)
    return best[1], best[2], best[3], True


def play_lookahead(seed, future_src="real", D=5, B=12, S=8, T=100, use_heur=True):
    """统一前瞻玩家。future_src 决定它"看到"什么未来，其余完全同构：
      'real'        = seer：那一条真实未来(信息满分；S 无效，单条)。
      'sampled_avg' = blind：S 份采样未来取平均(无真信息，但按分布最优边缘化)。
      'sampled_one' = antiseer：1 份采样未来当成真的(无真信息，单条 commit;
                       与 seer 结构对称 → seer-antiseer = 去掉'单条vs平均'不对称后的纯信息价值)。
    返回 (score, rounds_survived)。"""
    stream = _deal_stream(seed, T)
    board = 0; combo = 0; total = 0; rounds = 0; pos = 0
    for move in range(T):
        hand = stream[pos:pos + 3]
        if future_src == "real":
            fut = stream[pos + 3: pos + 3 + 3 * D]
            voc = lambda nb, nc, _f=fut: _rollout_fixed_strong(nb, nc, _f, B=B_ROLL)
        elif future_src == "sampled_one":
            r = random.Random(f"anti-{seed}-{move}")
            fut = [r.randrange(NUM_TYPES) for _ in range(3 * D)]
            voc = lambda nb, nc, _f=fut: _rollout_fixed_strong(nb, nc, _f, B=B_ROLL)
        else:  # sampled_avg
            futs = [[random.Random(f"blind-{seed}-{move}-{s}").randrange(NUM_TYPES)
                     for _ in range(3 * D)] for s in range(S)]
            voc = lambda nb, nc, _f=futs: sum(
                _rollout_fixed_strong(nb, nc, f, B=B_ROLL) for f in _f) / len(_f)
        hs, nb, nc, alive = _lookahead_step(board, combo, hand, B, voc, use_heur)
        if not alive:
            break
        total += hs; board = nb; combo = nc; rounds += 1; pos += 3
    return total, rounds


# 便捷别名
def play_seer(seed, D=5, B=12, T=100, use_heur=True):
    return play_lookahead(seed, "real", D, B, 1, T, use_heur)


def play_blind(seed, D=5, B=12, S=8, T=100, use_heur=True):
    return play_lookahead(seed, "sampled_avg", D, B, S, T, use_heur)


def play_antiseer(seed, D=5, B=12, T=100, use_heur=True):
    return play_lookahead(seed, "sampled_one", D, B, 1, T, use_heur)


# ---------- 实验 ----------
def _bootstrap_report(name, xs):
    pm, plo, phi = bootstrap_ci(xs, _median, boot_seed=f"med-{name}")
    mm, mlo, mhi = bootstrap_ci(xs, boot_seed=f"mean-{name}")
    print(f"  {name:<26} mean {mm:>7.0f} [{mlo:.0f},{mhi:.0f}]  "
          f"median {pm:>6.0f} [{plo:.0f},{phi:.0f}]", flush=True)
    return {"mean": mm, "mean_ci": [mlo, mhi], "median": pm, "median_ci": [plo, phi],
            "iqr": [pct(xs, 25), pct(xs, 75)]}


def d_sweep(seeds, Ds=(1, 3, 5, 8), B=12, T=80):
    """选 D：固定 horizon T 下，seer 存活率 + 分数随 D 是否 plateau。"""
    print(f"=== D-sweep (seer, fixed horizon T={T}, N={len(seeds)}) ===", flush=True)
    print(f"{'D':>3} {'surv@T':>7} {'score_med':>10} {'rounds_med':>11}", flush=True)
    out = {}
    for D in Ds:
        res = [play_seer(s, D=D, B=B, T=T) for s in seeds]
        sc = [r[0] for r in res]; rd = [r[1] for r in res]
        surv = sum(1 for r in rd if r >= T) / len(rd)
        print(f"{D:>3} {surv:>7.2f} {pct(sc,50):>10.0f} {pct(rd,50):>11.0f}", flush=True)
        out[D] = {"surv": surv, "score_med": pct(sc, 50), "rounds_med": pct(rd, 50)}
    return out


def survival_curve(seeds, D=5, B=12, S=8, Ts=(20, 40, 60, 80, 100, 150, 200)):
    """通道A 存活运气：各玩家活过 t 轮的比例(同一副牌跑到 max(Ts))。"""
    Tmax = max(Ts)
    print(f"\n=== survival curve (D={D}, N={len(seeds)}, Tmax={Tmax}) ===", flush=True)
    rounds = {"strong": [], "blind": [], "seer": []}
    for s in seeds:
        rounds["strong"].append(play_strong(s, T=Tmax, B=B)[1])
        rounds["blind"].append(play_blind(s, D=D, B=B, S=S, T=Tmax)[1])
        rounds["seer"].append(play_seer(s, D=D, B=B, T=Tmax)[1])
    print(f"  {'t':>5} " + " ".join(f"{p:>8}" for p in rounds), flush=True)
    curves = {p: [] for p in rounds}
    for t in Ts:
        row = {p: sum(1 for r in rounds[p] if r >= t) / len(seeds) for p in rounds}
        for p in rounds:
            curves[p].append(row[p])
        print(f"  {t:>5} " + " ".join(f"{row[p]:>8.2f}" for p in rounds), flush=True)
    # 杀手序列：报 seer 的【每轮死亡 hazard】(=死亡数/总在险轮数)，而非单一 T 的死亡率，
    # 避免结论是某个 T 的人为产物。真最优活得≥seer → 这是 hazard 的上界。
    seer_deaths = sum(1 for r in rounds["seer"] if r < Tmax)
    seer_at_risk = sum(min(r, Tmax) for r in rounds["seer"])
    hazard_point = seer_deaths / seer_at_risk if seer_at_risk else float("nan")
    hazard_ci95, mu_up = poisson_upper_95(seer_deaths, seer_at_risk)
    per_death = (1.0 / hazard_ci95) if hazard_ci95 else float("nan")
    # 采样 CI（点估计 vs 单侧 95% 上界）与"seer≤真最优"建模界是两回事，分开陈述。
    print(f"  -> seer per-round death hazard: POINT {hazard_point:.2e}/round "
          f"({seer_deaths} deaths / {seer_at_risk} at-risk rounds)", flush=True)
    print(f"     sampling 95% one-sided Poisson UPPER bound = {hazard_ci95:.2e}/round "
          f"(≤1 death per ~{per_death:.0f} rounds)", flush=True)
    print(f"     [separate MODELING bound] seer ≤ true-optimal → true killer-seq rate is even smaller",
          flush=True)
    print(f"     seer survival@Tmax={curves['seer'][-1]:.2f}; 全曲线见上(结论=整条曲线,非单点)",
          flush=True)
    return {"Ts": list(Ts), "curves": curves, "rounds": rounds,
            "seer_hazard_point_per_round": hazard_point,
            "seer_hazard_poisson_ub95_per_round": hazard_ci95,
            "seer_deaths": seer_deaths, "seer_at_risk": seer_at_risk,
            "seer_surv_at_Tmax": curves["seer"][-1]}


def channel_analysis(seeds, D=5, B=12, S=8, T=100):
    """通道B 计分运气(EVPI) + 带符号分解。固定 horizon T、CRN 配对。
    审核2 修正：死亡玩家分数被冻结 → 固定T总分会把存活漏进计分通道。故主指标用
    【四个玩家都活到 T 的 cohort】上的配对差；per-round 为次指标；固定T总分仅作三级参考。"""
    print(f"\n=== channel B: scoring/EVPI (D={D}, B={B}, S={S}, T={T}, N={len(seeds)}) ===",
          flush=True)
    P = {k: [] for k in ("strong", "blind", "seer", "anti")}
    R = {k: [] for k in P}
    for s in seeds:
        for k, fn in (("strong", lambda s: play_strong(s, T=T, B=B)),
                      ("blind", lambda s: play_blind(s, D=D, B=B, S=S, T=T)),
                      ("seer", lambda s: play_seer(s, D=D, B=B, T=T)),
                      ("anti", lambda s: play_antiseer(s, D=D, B=B, T=T))):
            sc, rd = fn(s); P[k].append(sc); R[k].append(rd)
    # cohort: headline 三玩家(strong/blind/seer)都活到 T 的种子(此子集分数=纯计分)。
    # anti 是诊断项且常死，不纳入 cohort 门槛，否则浪费种子(审核2:anti 仅诊断)。
    cohort = [i for i in range(len(seeds))
              if all(R[k][i] >= T for k in ("strong", "blind", "seer"))]
    anti_cohort = [i for i in cohort if R["anti"][i] >= T]
    print(f"  {'player':<10} {'score_med':>10} {'surv@T':>7}", flush=True)
    for k in ("strong", "blind", "anti", "seer"):
        print(f"  {k:<10} {pct(P[k],50):>10.0f} "
              f"{sum(1 for r in R[k] if r>=T)/len(seeds):>7.2f}", flush=True)
    print(f"  all-survive cohort: {len(cohort)}/{len(seeds)} seeds", flush=True)
    out = {"means": {k: sum(P[k]) / len(P[k]) for k in P},
           "medians": {k: pct(P[k], 50) for k in P},
           "surv": {k: sum(1 for r in R[k] if r >= T) / len(seeds) for k in P},
           "cohort_n": len(cohort)}
    if len(cohort) < 3:
        print("  !! cohort too small for CI — lower T or raise N", flush=True)
        return out
    if len(cohort) < 20:
        print(f"  !! WARN cohort={len(cohort)}<20 — CIs unreliable; raise N or lower T", flush=True)

    def coh(k):
        return [P[k][i] for i in cohort]
    print(f"  --- PRIMARY: paired gaps on all-survive cohort (n={len(cohort)}) ---", flush=True)
    g_raw = [a - b for a, b in zip(coh("seer"), coh("strong"))]
    g_evpi = [a - b for a, b in zip(coh("seer"), coh("blind"))]
    g_proc = [a - b for a, b in zip(coh("blind"), coh("strong"))]
    out["raw_gap_seer_strong"] = _bootstrap_report("raw gap (seer-strong)", g_raw)
    out["EVPI_vs_marginalizer"] = _bootstrap_report("EVPI [HEADLINE](seer-blind)", g_evpi)
    out["procedure_cost"] = _bootstrap_report("procedure (blind-strong)", g_proc)
    # anti 诊断项：仅在 anti 也存活的子集上算(否则混入存活差异)
    if len(anti_cohort) >= 3:
        ac = lambda k: [P[k][i] for i in anti_cohort]
        out["EVPI_clean_diag"] = _bootstrap_report(
            f"EVPI_clean DIAG n={len(anti_cohort)}",
            [a - b for a, b in zip(ac("seer"), ac("anti"))])
        out["single_draw_bias_diag"] = _bootstrap_report(
            "1-draw bias DIAG(anti-bl)",
            [a - b for a, b in zip(ac("anti"), ac("blind"))])
    # 带符号堆叠分解：raw = EVPI(seer-blind) + procedure(blind-strong)。不 clip。
    print("  --- signed decomposition (means, cohort) ---", flush=True)
    em = sum(g_evpi) / len(g_evpi); pm = sum(g_proc) / len(g_proc)
    rm = sum(g_raw) / len(g_raw)
    print(f"  raw {rm:.0f} = EVPI(info) {em:.0f} + procedure(search) {pm:.0f}", flush=True)
    # 份额：per-seed 配对中位数 (seer-blind)/(seer-strong)，仅取分母>0 的种子。
    # 注：denom>0 过滤使份额**条件于 seer>strong 的种子**(会上偏)，须显式声明 + 报 n_valid/n_cohort。
    shares = [(se - bl) / (se - st) for se, bl, st in
              zip(coh("seer"), coh("blind"), coh("strong")) if se - st > 0]
    if shares:
        sm, slo, shi = bootstrap_ci(shares, _median, boot_seed="evpi-share")
        print(f"  EVPI/raw per-seed median share = {sm*100:.0f}% "
              f"[{slo*100:.0f},{shi*100:.0f}]  (n_valid={len(shares)}/{len(cohort)} cohort; "
              f"**conditional on seer>strong**; 余下为搜索功劳/代价)", flush=True)
        out["evpi_share_per_seed_median"] = sm
        out["evpi_share_ci"] = [slo, shi]
        out["evpi_share_n_valid"] = len(shares)
    else:
        out["evpi_share_per_seed_median"] = float("nan")
    # 次指标：per-round-survived rate（cohort 上等价于 /T，但留给非cohort全局视角）
    out["per_round"] = {k: sum(P[k][i] / max(R[k][i], 1) for i in range(len(seeds)))
                        / len(seeds) for k in P}
    print(f"  [secondary] per-round score: " +
          " ".join(f"{k} {out['per_round'][k]:.1f}" for k in
                   ("strong", "blind", "seer")), flush=True)
    return out


def s_stability(seeds, D=5, B=12, T=80, Ss=(4, 8, 16, 32)):
    """审核2/3/4: 验证 blind 的 S 够大(EVPI 在 S 上趋平)。
    审核4 修正：旧版各 S 用各自存活 cohort(n=19/23/28/25) → 非 apples-to-apples。
    改为**固定 cohort = 全 S 的 intersection-of-survivors**(seer 存活 ∧ ∀S blind(S) 存活)，
    在同一组种子上比各 S 的 EVPI_med。若 fixed cohort <20，警告须升 N 或降 T。
    blind 的 RNG 用 'blind-{seed}-{move}-{s}'，改 offset 即换一组独立样本。"""
    print(f"\n=== S-stability (EVPI=seer-blind vs S, FIXED intersection cohort, "
          f"T={T}, N={len(seeds)}) ===", flush=True)
    seer = {s: play_seer(s, D=D, B=B, T=T) for s in seeds}
    blind = {S: {s: play_blind(s, D=D, B=B, S=S, T=T) for s in seeds} for S in Ss}
    # 固定 cohort：seer 活到 T 且 每个 S 的 blind 都活到 T
    cohort = [s for s in seeds
              if seer[s][1] >= T and all(blind[S][s][1] >= T for S in Ss)]
    print(f"  fixed intersection cohort = {len(cohort)}/{len(seeds)} seeds", flush=True)
    if len(cohort) < 20:
        print(f"  !! WARN fixed cohort={len(cohort)}<20 — raise N or lower T for reliable CIs",
              flush=True)
    print(f"  {'S':>4} {'EVPI_med':>9} {'cohort':>7}", flush=True)
    out = {"fixed_cohort_n": len(cohort)}
    for S in Ss:
        gaps = [seer[s][0] - blind[S][s][0] for s in cohort]
        med = pct(gaps, 50) if gaps else float("nan")
        print(f"  {S:>4} {med:>9.0f} {len(gaps):>7}", flush=True)
        out[S] = {"evpi_med": med, "cohort_n": len(gaps)}
    return out


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "probe"
    N = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    seeds = list(range(N))
    import json
    if mode == "probe":
        import time
        for D in (3, 5):
            t = time.time(); sc = [play_seer(s, D=D, T=40)[0] for s in range(3)]
            print(f"D={D} T40: 3 seer in {time.time()-t:.1f}s scores={[f'{x:.0f}' for x in sc]}", flush=True)
    elif mode == "sweep":
        T = int(sys.argv[3]) if len(sys.argv) > 3 else 80
        json.dump(d_sweep(seeds, T=T), open("sweep.json", "w"), indent=2)
    elif mode == "survival":
        D = int(sys.argv[3]) if len(sys.argv) > 3 else 5
        json.dump(survival_curve(seeds, D=D), open("survival.json", "w"), indent=2)
    elif mode == "channel":
        D = int(sys.argv[3]) if len(sys.argv) > 3 else 5
        # 审核2: EVPI 份额在 ≥2 个 T 报告，证明不是单一 T 的人为产物
        Ts = [int(x) for x in sys.argv[4].split(",")] if len(sys.argv) > 4 else [80, 120]
        allR = {}
        for T in Ts:
            allR[f"T{T}"] = channel_analysis(seeds, D=D, T=T)
        json.dump({"N": N, "D": D, "Ts": Ts, **allR}, open("channelB.json", "w"), indent=2)
        print("\n-> wrote channelB.json", flush=True)
    elif mode == "sstab":
        D = int(sys.argv[3]) if len(sys.argv) > 3 else 5
        T = int(sys.argv[4]) if len(sys.argv) > 4 else 40
        json.dump(s_stability(seeds, D=D, T=T), open("sstab.json", "w"), indent=2)
