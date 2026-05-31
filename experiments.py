"""
最终实验：统一用 fast.py(bitboard) 引擎产出 writeup 所需全部数字，写入 results.json。
四块：
  1) 方差分解 (random/greedy/strong) — 技能 vs 运气 vs 交互
  2) 技能阶梯 (ε-greedy random↔strong) + 缺口/σ 交叉点
  3) 配对收敛 (strong / LA_greedy / LA_strongbase) — CRN 同种子
  4) oracle 缺口 (hindsight 上界) — 运气=信息价值 的量化
"""
import json
import random
import sys
from fast import NUM_TYPES, greedy_hand, strong_hand, make_lookahead
from run_lookahead import greedy_random, play_eps, play, play_oracle


def stat(xs):
    m = sum(xs) / len(xs)
    sd = (sum((x - m) ** 2 for x in xs) / len(xs)) ** 0.5
    s = sorted(xs)
    def pc(p):
        k = (len(s) - 1) * p / 100; lo = int(k); hi = min(lo + 1, len(s) - 1)
        return s[lo] + (s[hi] - s[lo]) * (k - lo)
    return {"mean": m, "std": sd, "min": min(xs), "max": max(xs),
            "p10": pc(10), "p50": pc(50), "p90": pc(90)}


def play_simple(policy, seed):
    deal = random.Random(f"deal-{seed}")
    act = random.Random(f"act-{seed}")
    board = 0; combo = 0; total = 0
    while True:
        hand = [deal.randrange(NUM_TYPES) for _ in range(3)]
        pts, board, combo, alive = policy(board, combo, hand, act)
        total += pts
        if not alive:
            return total


def variance_decomp(scores_by_policy):
    names = list(scores_by_policy)
    S = len(scores_by_policy[names[0]]); P = len(names)
    allv = [scores_by_policy[p][s] for p in names for s in range(S)]
    grand = sum(allv) / len(allv)
    mp = {p: sum(scores_by_policy[p]) / S for p in names}
    ms = [sum(scores_by_policy[p][s] for p in names) / P for s in range(S)]
    sst = sum((x - grand) ** 2 for x in allv)
    ssp = S * sum((mp[p] - grand) ** 2 for p in names)
    sss = P * sum((m - grand) ** 2 for m in ms)
    return {"skill": ssp / sst, "luck": sss / sst, "inter": (sst - ssp - sss) / sst}


if __name__ == "__main__":
    NV = int(sys.argv[1]) if len(sys.argv) > 1 else 200   # 方差/阶梯种子
    NP = int(sys.argv[2]) if len(sys.argv) > 2 else 24    # 配对/oracle 种子(慢)
    R = {"config": {"n_var": NV, "n_paired": NP, "piece_types": NUM_TYPES}}

    # 1) 方差分解
    print("[1/4] 方差分解...")
    sc = {"random": [], "greedy": [], "strong": []}
    for s in range(NV):
        sc["random"].append(play_simple(greedy_random, s))
        sc["greedy"].append(play_simple(greedy_hand, s))
        sc["strong"].append(play_simple(strong_hand, s))
    R["players"] = {p: stat(v) for p, v in sc.items()}
    R["decomp_all"] = variance_decomp(sc)
    R["decomp_skilled"] = variance_decomp({k: sc[k] for k in ("greedy", "strong")})

    # 2) 技能阶梯
    print("[2/4] 技能阶梯...")
    eps_list = [1.0, 0.8, 0.6, 0.4, 0.2, 0.0]
    ladder = []
    for eps in eps_list:
        xs = [play_eps(s, eps) for s in range(NV)]
        st = stat(xs); st["skill_axis"] = 1 - eps; ladder.append(st)
    ceiling = max(r["mean"] for r in ladder)
    R["ceiling"] = ceiling
    for r in ladder:
        r["gap"] = ceiling - r["mean"]
        r["ratio"] = r["gap"] / r["std"] if r["std"] else 0
        r["dominant"] = "skill" if r["gap"] > r["std"] else "luck"
    R["ladder"] = ladder
    rs = sorted(ladder, key=lambda r: r["mean"]); cross = None
    for i in range(len(rs) - 1):
        d1 = ceiling - rs[i]["mean"] - rs[i]["std"]
        d2 = ceiling - rs[i + 1]["mean"] - rs[i + 1]["std"]
        if d1 > 0 >= d2:
            t = d1 / (d1 - d2); cross = rs[i]["mean"] + t * (rs[i + 1]["mean"] - rs[i]["mean"]); break
    R["crossover"] = cross

    # 3) 配对收敛
    print("[3/4] 配对收敛 (慢)...")
    la_g = make_lookahead(D=3, S=8, B=10, base="greedy")
    la_s = make_lookahead(D=2, S=4, B=8, base="strong")
    P_strong, P_lag, P_las = [], [], []
    for s in range(NP):
        P_strong.append(play_simple(strong_hand, s))
        P_lag.append(play(la_g, s, needs_ctx=True))
        P_las.append(play(la_s, s, needs_ctx=True))
    def paired(a, b):
        d = [x - y for x, y in zip(a, b)]
        sd = (sum((x - sum(d)/len(d))**2 for x in d)/len(d))**0.5
        return {"mean_diff": sum(d) / len(d), "se": sd / len(d) ** 0.5,
                "wins": sum(1 for x in d if x > 0), "n": len(d)}
    R["paired"] = {
        "strong": stat(P_strong), "la_greedy": stat(P_lag), "la_strongbase": stat(P_las),
        "la_greedy_vs_strong": paired(P_lag, P_strong),
        "la_strongbase_vs_strong": paired(P_las, P_strong),
        "la_strongbase_vs_la_greedy": paired(P_las, P_lag)}

    # 4) oracle 缺口 —— 【已撤回】旧 ratio-of-means 口径(beam-rollout seer 近乎不死→分差
    #    被存活长度主导、本质无界)。新口径=固定 horizon 两通道，见 oracle_analysis.py。
    R["oracle_RETRACTED"] = ("旧 78% 缺口口径已撤回。新口径见 oracle_analysis.py: "
                             "通道A 存活 hazard(survival.json) + 通道B EVPI(channelB.json)。")

    with open("results.json", "w") as f:
        json.dump(R, f, indent=2, ensure_ascii=False)
    print("\n写入 results.json")
    print(f"  方差分解(熟练间): 技能{R['decomp_skilled']['skill']*100:.0f}% "
          f"运气{R['decomp_skilled']['luck']*100:.0f}% 交互{R['decomp_skilled']['inter']*100:.0f}%")
    print(f"  交叉点≈{R['crossover']:.0f}  天花板≈{ceiling:.0f}")
    print("  oracle 缺口已迁至 oracle_analysis.py (两通道, 见 README)")
    print(f"  收敛: LA_strongbase−strong={R['paired']['la_strongbase_vs_strong']['mean_diff']:+.0f}"
          f"±{R['paired']['la_strongbase_vs_strong']['se']:.0f}")
