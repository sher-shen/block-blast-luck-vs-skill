"""
evpi_vla.py — 预注册 verdict (i) 落地：用最强可玩在线策略 vla(价值引导前瞻, ~2924)
当 EVPI 占比的新分母，在冻结 intersection-of-survivors cohort 上重算"运气=信息价值"占比。

背景(见 memory/oracle_immortality_reframe.md, ROADMAP 预注册结果表):
  原 channelB.json 把休闲玩家的"到 oracle 的缺口"拆成
    raw(seer-strong) = EVPI(seer-blind) + procedure(blind-strong)
  占比 = EVPI/raw 的 per-seed 配对中位数 = 69/65/57% (T=40/50/60)。
  ROADMAP 风险(d)+预注册 verdict(i): 分母(seer-strong)载重于"strong=在线天花板"这个假设；
  更强在线玩家会**抬升占比**(占比对在线强度是下偏保守)。RL/vla(2924) ≫ strong(2364)
  ⇒ 命中 verdict(i)：用 vla 当新分母在同一冻结 cohort 上重算，占比只会升。

本脚本(诚实记录关键口径):
  - 三 oracle 玩家(strong/blind/seer)用 oracle_analysis 的**原函数**重跑(确定性,种子相同
    ⇒ 必复现 channelB 的 cohort 与 69/65/57% 占比 = 自洽 sanity check)。D_oracle=3, N=120
    匹配 channelB.json。
  - vla = 交付的最强可玩策略(价值引导前瞻 D2 S30 B12 + 训练 V, models/strongest_v.pt)，
    喂**同一 _deal_stream(seed)** ⇒ 与 strong/blind/seer 严格 CRN 配对。
    T_MAX=50 是网络 k 归一化常数(随模型固定)，三个 horizon 都用它；T=60 前若干轮 k_norm>1 = 轻度 OOD，已标注。
  - cohort 选择:
      cohort3 = {strong,blind,seer 都活到 T}  (channelB 原 cohort，复现用)
      cohort4 = {strong,blind,seer,vla 都活到 T} (vla-inclusive 冻结 cohort，vla 占比主口径)
    在 cohort4 上同时报 strong-分母 与 vla-分母占比 = apples-to-apples 隔离"换分母"效应(非 cohort 变化)。
  - 带符号分解(不 clip): raw_vla(seer-vla) = EVPI(seer-blind) + procedure_vla(blind-vla)。
    若 vla>blind 则 procedure_vla<0 ⇒ 占比可 >100%，含义=vla 本身已是比 blind 更强的边缘化器，
    seer-vla 残差几乎纯信息价值(运气)，procedure 技能通道已被 vla 吃满。诚实报，不掩盖。

零新建模假设；只换在线天花板。print 全部 flush=True。
"""
import json
import sys
import time
from multiprocessing import Pool

import oracle_analysis as oa

D_ORACLE = 3
N_DEFAULT = 120
TS = [40, 50, 60]
TMAX_MODEL = 50          # vla 网络 k 归一化常数(训练 horizon)，随模型固定
VLA_D, VLA_S, VLA_B = 2, 30, 12
VLA_CKPT = "models/strongest_v.pt"
VLA_HIDDEN = 128
VLA_TRANSFORM = "rate"


# ---------- oracle 三玩家：纯 python，多进程并行(worker 只 import oracle_analysis，不碰 torch) ----------
def _oracle_worker(arg):
    seed, T = arg
    import oracle_analysis as oa
    st_s, st_r = oa.play_strong(seed, T=T)
    bl_s, bl_r = oa.play_blind(seed, D=D_ORACLE, T=T)
    se_s, se_r = oa.play_seer(seed, D=D_ORACLE, T=T)
    return (seed, T, st_s, st_r, bl_s, bl_r, se_s, se_r)


def run_oracle(seeds):
    tasks = [(s, T) for T in TS for s in seeds]
    print(f"[oracle] {len(tasks)} (seed,T) tasks across pool ...", flush=True)
    t = time.time()
    res = {T: {} for T in TS}
    with Pool() as pool:
        for (seed, T, ss, sr, bs, br, es, er) in pool.imap_unordered(_oracle_worker, tasks, chunksize=4):
            res[T][seed] = {"strong": (ss, sr), "blind": (bs, br), "seer": (es, er)}
    print(f"[oracle] done in {time.time()-t:.0f}s", flush=True)
    return res


# ---------- vla：GPU，单进程顺序(每 horizon 单独跑，k-条件 ⇒ horizon-aware) ----------
def run_vla(seeds):
    import rl8
    rl8.T_MAX = TMAX_MODEL
    eng = rl8.Engine8()
    net = rl8.RateNet(hidden=VLA_HIDDEN).to(rl8.DEVICE)
    net.load_state_dict(rl8.torch.load(VLA_CKPT, map_location=rl8.DEVICE))
    net.eval()
    print(f"[vla] ckpt={VLA_CKPT} D={VLA_D} S={VLA_S} B={VLA_B} T_MAX={TMAX_MODEL} "
          f"DEVICE={rl8.DEVICE}", flush=True)
    res = {T: {} for T in TS}
    for T in TS:
        t = time.time()
        for s in seeds:
            flat = oa._deal_stream(s, T)
            stream = [flat[3 * i:3 * i + 3] for i in range(T)]
            sc, rd = rl8.play_vlookahead_stream(net, eng, stream, VLA_TRANSFORM,
                                                VLA_D, VLA_S, VLA_B, seed_idx=s)
            res[T][s] = (sc, rd)
        print(f"[vla] T={T}: {len(seeds)} games in {time.time()-t:.0f}s", flush=True)
    return res


# ---------- 占比统计(沿用 oracle_analysis 的 bootstrap/pct，口径一致) ----------
def _paired_share(seer, ref, blind, cohort):
    """per-seed (seer-blind)/(seer-ref) 中位数 + bootstrap CI；仅取分母>0。返回 dict。"""
    vals = [(seer[i] - blind[i]) / (seer[i] - ref[i])
            for i in cohort if seer[i] - ref[i] > 0]
    if not vals:
        return {"median": float("nan"), "ci": [float("nan")] * 2, "n_valid": 0, "n_cohort": len(cohort)}
    m, lo, hi = oa.bootstrap_ci(vals, oa._median, boot_seed="vla-share")
    return {"median": m, "ci": [lo, hi], "n_valid": len(vals), "n_cohort": len(cohort)}


def _gap_stats(xs):
    m, lo, hi = oa.bootstrap_ci(xs, oa._median, boot_seed="gap")
    mn = sum(xs) / len(xs)
    return {"mean": mn, "median": m, "median_ci": [lo, hi]}


def analyze(oracle, vla, seeds):
    out = {"N": len(seeds), "D_oracle": D_ORACLE, "Ts": TS,
           "vla_cfg": {"D": VLA_D, "S": VLA_S, "B": VLA_B, "ckpt": VLA_CKPT,
                       "hidden": VLA_HIDDEN, "T_MAX": TMAX_MODEL}}
    for T in TS:
        o = oracle[T]; v = vla[T]
        sc = {k: {s: o[s][k][0] for s in seeds} for k in ("strong", "blind", "seer")}
        sc["vla"] = {s: v[s][0] for s in seeds}
        rd = {k: {s: o[s][k][1] for s in seeds} for k in ("strong", "blind", "seer")}
        rd["vla"] = {s: v[s][1] for s in seeds}
        surv = {k: sum(1 for s in seeds if rd[k][s] >= T) / len(seeds) for k in sc}

        cohort3 = [s for s in seeds if all(rd[k][s] >= T for k in ("strong", "blind", "seer"))]
        cohort4 = [s for s in seeds if all(rd[k][s] >= T for k in ("strong", "blind", "seer", "vla"))]
        vla_dead_in_c3 = sum(1 for s in cohort3 if rd["vla"][s] < T)

        def arr(k, coh):
            return {i: sc[k][s] for i, s in enumerate(coh)}, [sc[k][s] for s in coh]

        # --- cohort3: 复现 channelB 的 strong-分母占比 (sanity) ---
        c3 = {k: [sc[k][s] for s in cohort3] for k in sc}
        idx3 = list(range(len(cohort3)))
        share_strong_c3 = _paired_share(c3["seer"], c3["strong"], c3["blind"], idx3)

        # --- cohort4: vla-分母占比(主) + 同 cohort 上的 strong-分母占比(隔离换分母) ---
        c4 = {k: [sc[k][s] for s in cohort4] for k in sc}
        idx4 = list(range(len(cohort4)))
        share_strong_c4 = _paired_share(c4["seer"], c4["strong"], c4["blind"], idx4)
        share_vla_c4 = _paired_share(c4["seer"], c4["vla"], c4["blind"], idx4)

        # 带符号分解(均值, cohort4): raw_vla = EVPI + procedure_vla
        raw_vla = [c4["seer"][i] - c4["vla"][i] for i in idx4]
        evpi = [c4["seer"][i] - c4["blind"][i] for i in idx4]
        proc_vla = [c4["blind"][i] - c4["vla"][i] for i in idx4]
        raw_strong = [c4["seer"][i] - c4["strong"][i] for i in idx4]

        out[f"T{T}"] = {
            "surv": surv,
            "score_median_full": {k: oa.pct([sc[k][s] for s in seeds], 50) for k in sc},
            "cohort3_n": len(cohort3), "cohort4_n": len(cohort4),
            "vla_deaths_in_cohort3": vla_dead_in_c3,
            "score_median_cohort4": {k: oa.pct(c4[k], 50) for k in sc},
            "share_strong_denom_cohort3_SANITY": share_strong_c3,
            "share_strong_denom_cohort4": share_strong_c4,
            "share_vla_denom_cohort4_PRIMARY": share_vla_c4,
            "raw_gap_seer_strong_cohort4": _gap_stats(raw_strong),
            "raw_gap_seer_vla_cohort4": _gap_stats(raw_vla),
            "EVPI_seer_blind_cohort4": _gap_stats(evpi),
            "procedure_vla_blind_minus_vla_cohort4": _gap_stats(proc_vla),
        }
        print(f"\n=== T={T} (cohort3={len(cohort3)}, cohort4={len(cohort4)}, "
              f"vla deaths in c3={vla_dead_in_c3}) ===", flush=True)
        print(f"  surv: " + " ".join(f"{k}={surv[k]:.2f}" for k in sc), flush=True)
        print(f"  median score (cohort4): " +
              " ".join(f"{k}={oa.pct(c4[k],50):.0f}" for k in ("strong", "blind", "vla", "seer")),
              flush=True)
        print(f"  share strong-denom  cohort3 (SANITY vs channelB)= "
              f"{share_strong_c3['median']*100:.0f}% "
              f"[{share_strong_c3['ci'][0]*100:.0f},{share_strong_c3['ci'][1]*100:.0f}] "
              f"n={share_strong_c3['n_valid']}/{share_strong_c3['n_cohort']}", flush=True)
        print(f"  share strong-denom  cohort4 = {share_strong_c4['median']*100:.0f}% "
              f"[{share_strong_c4['ci'][0]*100:.0f},{share_strong_c4['ci'][1]*100:.0f}]", flush=True)
        print(f"  share VLA-denom     cohort4 = {share_vla_c4['median']*100:.0f}% "
              f"[{share_vla_c4['ci'][0]*100:.0f},{share_vla_c4['ci'][1]*100:.0f}]  <-- PRE-REG verdict(i)",
              flush=True)
        em = sum(evpi) / len(evpi); pm = sum(proc_vla) / len(proc_vla); rm = sum(raw_vla) / len(raw_vla)
        print(f"  decomp(mean): raw_vla {rm:.0f} = EVPI {em:.0f} + procedure_vla {pm:.0f} "
              f"({'vla>blind -> procedure absorbed' if pm < 0 else 'blind>vla'})", flush=True)
    return out


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    N = int(sys.argv[1]) if len(sys.argv) > 1 else N_DEFAULT
    seeds = list(range(N))
    print(f"=== EVPI recompute with vla denominator | N={N} D_oracle={D_ORACLE} "
          f"Ts={TS} ===", flush=True)
    t0 = time.time()
    oracle = run_oracle(seeds)
    vla = run_vla(seeds)
    out = analyze(oracle, vla, seeds)
    json.dump(out, open("evpi_vla.json", "w"), indent=2)
    print(f"\n-> wrote evpi_vla.json  (total {time.time()-t0:.0f}s)", flush=True)
