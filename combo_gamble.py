"""
combo_gamble.py — "连击豪赌"计算器。把 Block Blast 的连击当作一场**press-your-luck 赌局**量化。

为什么连击是赌局：
  连击计数 combo 在**每次放置**触发消除(消≥1条)时 +1，任一次放置没消除则**清零**。
  奖励随连击层级**超线性**上涨(assumed 档: +50×(streak-1)/次消; real_approx 档: clear bonus ×(1+(streak-1)×0.5))。
  ⇒ 维持长连击赔率诱人，但一次维持失败前功尽弃 = 经典"要不要继续押"结构。
  这条**超线性 combo 正是本游戏高方差(运气重)的物理来源** —— 直接服务于 luck/skill 主线，
  也是"最强策略为何要刻意保连击"的可玩性洞察。

两部分：
  (A) 解析计算器(精确、零依赖)：把连击建模为"每步以维持概率 p 续命的几何赌局"。
      层级 s 的一次消除给 combo 价值 u·(s-1)（assumed: u=combo_unit=50；mult: u=clear_points(L̄)·combo_mult）。
      正在 C 层、之后每步维持概率 p 的**期望未来 combo 奖励**：
        G(C,p) = u · p/(1-p) · [ (C-1) + 1/(1-p) ]        (闭式，对 j≥1 求 p^j·u·(C+j-1))
      豪赌决策量(只看 combo 项)：在岔路"保连击(强消, C→C+1)"vs"断连击(不消, →0)"：
        ΔG(C,p) = [ u·C + G(C+1,p) ] − G(0,p)
      这是你为保住连击愿意付出的"棋盘健康度代价"上限。给定每次强消的棋盘代价 cost，
      保连击 iff ΔG(C,p) > cost；可反解 break-even p 或 break-even cost。

  (B) 经验标定(用真实模拟把 p / L̄ / combo 占分比落地，非空想)：
      用 beam_hand_path 复刻 strong 策略的逐块落点 → 逐次放置重放 score_placement，
      记录每次放置是否消除、连击层级、combo 奖励 → 估计
        p_maintain(整体 + 按当前层级条件化)、streak 长度分布、combo 奖励占总分比、L̄(平均每消条数)。
  纯标准库 SVG 出图。print 全部 flush=True。
"""
import json
import sys
import time
from multiprocessing import Pool

import fast
from scoring import assumed, real_approx, score_placement


# ============================== (A) 解析计算器 ==============================
def combo_unit_value(scoring, Lbar):
    """每升一层连击、一次消除带来的 combo 增量价值 u。
    assumed(additive): u = combo_unit（与消除条数无关）。
    real_approx(mult): combo 把 clear bonus 乘 (1+(s-1)·mult) ⇒ 每层增量 = clear_points(L̄)·mult。"""
    if scoring.combo_mode == "mult":
        return scoring.clear_points(max(1, round(Lbar))) * scoring.combo_mult
    return float(scoring.combo_unit)


def G(C, p, u):
    """正在 C 层、之后每步维持概率 p 的期望未来 combo 奖励(几何闭式)。C≥0；p∈[0,1)。"""
    if p <= 0:
        return 0.0
    if p >= 1:
        p = 0.999999
    return u * p / (1 - p) * ((C - 1) + 1.0 / (1 - p))


def preserve_minus_break(C, p, u):
    """岔路上"保连击 vs 断连击"的纯 combo EV 增益 ΔG(C,p)。= 愿付棋盘代价上限。"""
    return (u * C + G(C + 1, p, u)) - G(0, p, u)


def G_levelcond(C, p_by_level, u, jmax=40):
    """诚实版：用**按层级条件化**的维持概率 p_c(实测会随层级衰减)而非常数 p。
    正在 C 层，到 C+j 层的概率 = ∏_{m=0}^{j-1} p_{C+m}；超出实测层级用最后一档外推。
    G_emp(C) = Σ_{j≥1} (∏ p) · u·(C+j-1)。常数-p 模型会高估真实赌局价值。"""
    if not p_by_level:
        return float("nan")
    last = max(p_by_level)
    tail = p_by_level[last]

    def p_at(c):
        return p_by_level.get(c, tail)
    total = 0.0
    prob = 1.0
    for j in range(1, jmax + 1):
        prob *= p_at(C + j - 1)
        if prob < 1e-9:
            break
        total += prob * u * (C + j - 1)
    return total


def preserve_minus_break_levelcond(C, p_by_level, u):
    """level-conditional 版 ΔG：保连击(到 C+1)−断连击(回 0 重建)。"""
    return (u * C + G_levelcond(C + 1, p_by_level, u)) - G_levelcond(0, p_by_level, u)


def breakeven_p(C, u, cost, lo=1e-4, hi=0.999999):
    """给定保连击的棋盘代价 cost，二分解出 ΔG(C,p)=cost 的 break-even 维持概率 p*。
    p>p* 才值得为保连击付该代价。无解(全程<cost 或 >cost)返回边界标记。"""
    if preserve_minus_break(C, hi, u) < cost:
        return None  # 即便 p→1 也不值
    if preserve_minus_break(C, lo, u) >= cost:
        return 0.0   # 任何 p 都值
    for _ in range(60):
        mid = (lo + hi) / 2
        if preserve_minus_break(C, mid, u) >= cost:
            hi = mid
        else:
            lo = mid
    return (lo + hi) / 2


def calc_report(p_emp=None, Lbar=2.0):
    """打印两套计分档下的计算器表 + 若给经验 p 则给出实测点的决策。"""
    out = {"Lbar": Lbar, "schemes": {}}
    for name, sc in (("assumed_additive", assumed()), ("real_approx_mult", real_approx())):
        u = combo_unit_value(sc, Lbar)
        print(f"\n=== 计算器 | {name} | u(每层 combo 增量价值)={u:.0f} (L̄={Lbar}) ===", flush=True)
        print(f"  {'p':>5} | " + " ".join(f"G(C={C})".rjust(9) for C in (1, 3, 5)) +
              " | " + " ".join(f"ΔG(C={C})".rjust(9) for C in (1, 3, 5)), flush=True)
        rows = {}
        for p in (0.3, 0.5, 0.7, 0.8, 0.9):
            gs = {C: G(C, p, u) for C in (1, 3, 5)}
            dg = {C: preserve_minus_break(C, p, u) for C in (1, 3, 5)}
            rows[p] = {"G": gs, "dG": dg}
            print(f"  {p:>5.2f} | " + " ".join(f"{gs[C]:>9.0f}" for C in (1, 3, 5)) +
                  " | " + " ".join(f"{dg[C]:>9.0f}" for C in (1, 3, 5)), flush=True)
        # break-even p：要为"保连击在 C=3"付棋盘代价 cost 时，需要的维持概率
        be = {cost: breakeven_p(3, u, cost) for cost in (20, 50, 100, 200)}
        print("  break-even 维持概率 p*(C=3, 付棋盘代价 cost 才值): " +
              " ".join(f"cost{c}->{'never' if be[c] is None else ('always' if be[c]==0.0 else f'{be[c]:.2f}')}"
                       for c in (20, 50, 100, 200)), flush=True)
        scheme = {"u": u, "rows": {str(p): rows[p] for p in rows},
                  "breakeven_p_C3": {str(c): be[c] for c in be}}
        if p_emp is not None:
            dgC = {C: preserve_minus_break(C, p_emp, u) for C in (1, 3, 5)}
            print(f"  >> 实测 p={p_emp:.2f} 处 ΔG: C=1->{dgC[1]:.0f}  C=3->{dgC[3]:.0f}  "
                  f"C=5->{dgC[5]:.0f}  (棋盘代价低于此则保连击+EV)", flush=True)
            scheme["at_empirical_p"] = {"p": p_emp, "dG": dgC}
        out["schemes"][name] = scheme
    return out


# ============================== (B) 经验标定：instrument strong ==============================
def _deal_stream(seed, T):
    import random
    deal = random.Random(f"deal-{seed}")
    return [deal.randrange(fast.NUM_TYPES) for _ in range(3 * (T + 16))]


def _instrument_strong_one(arg):
    """复刻 strong 玩一局(CRN deal-{seed})，逐次放置记录连击事件。返回紧凑统计。"""
    seed, T, B = arg
    import fast
    from scoring import score_placement
    SC = fast.SCORING
    stream = _deal_stream(seed, T)
    board, combo, total, pos = 0, 0, 0.0, 0
    n_place = 0           # 总放置次数
    n_clear = 0           # 触发消除的放置次数
    sum_L = 0             # 消除条数之和(用于 L̄)
    combo_bonus_total = 0.0
    streaks = []          # 每条连击链的长度(连续消除次数)
    # 按当前层级条件化的维持计数：placements_at_level[c] / maintained_at_level[c]
    at_level = {}         # c -> 在 combo=c 状态下做出的"会消除否"放置数
    maint_level = {}      # c -> 其中确实消除(维持)的数
    cur_streak = 0
    for _ in range(T):
        hand = stream[pos:pos + 3]
        cands = fast.beam_hand_path(board, combo, hand, B)
        if not cands:
            break
        # strong 的选择：score + heuristic 最大
        nb_c, nc_c, sc_c, path = max(cands, key=lambda s: s[2] + fast.heuristic(s[0]))
        # 逐次放置重放，提取每次放置的连击事件
        brun, crun = board, combo
        for (hidx, mask) in path:
            pid = hand[hidx]
            _, cl, empty = fast.apply_mask(brun, mask)
            pts, ncb = score_placement(SC, fast.NCELLS[pid], cl, empty, crun)
            cleared = cl > 0
            n_place += 1
            # 条件化维持统计：从"当前 crun 层"出发这次是否维持
            at_level[crun] = at_level.get(crun, 0) + 1
            if cleared:
                maint_level[crun] = maint_level.get(crun, 0) + 1
                n_clear += 1
                sum_L += cl
                cur_streak += 1
                # combo 奖励(超出"首消基础"的部分)：用 streak=1 同条件作差，口径与 scoring 一致
                base_pts, _ = score_placement(SC, fast.NCELLS[pid], cl, empty, 0)
                combo_bonus_total += (pts - base_pts)
            else:
                if cur_streak > 0:
                    streaks.append(cur_streak)
                cur_streak = 0
            brun, crun = fast.apply_mask(brun, mask)[0], ncb
        board, combo, total = nb_c, nc_c, total + sc_c
        pos += 3
    if cur_streak > 0:
        streaks.append(cur_streak)
    return {"seed": seed, "total": total, "n_place": n_place, "n_clear": n_clear,
            "sum_L": sum_L, "combo_bonus_total": combo_bonus_total, "streaks": streaks,
            "at_level": at_level, "maint_level": maint_level}


def empirical(N=120, T=50, B=12):
    print(f"=== 经验标定 strong | N={N} T={T} B={B} ===", flush=True)
    t = time.time()
    tasks = [(s, T, B) for s in range(N)]
    with Pool() as pool:
        res = list(pool.imap_unordered(_instrument_strong_one, tasks, chunksize=4))
    print(f"  done in {time.time()-t:.0f}s", flush=True)
    n_place = sum(r["n_place"] for r in res)
    n_clear = sum(r["n_clear"] for r in res)
    sum_L = sum(r["sum_L"] for r in res)
    combo_bonus = sum(r["combo_bonus_total"] for r in res)
    total = sum(r["total"] for r in res)
    p_maintain = n_clear / n_place if n_place else float("nan")
    Lbar = sum_L / n_clear if n_clear else float("nan")
    combo_share = combo_bonus / total if total else float("nan")
    # 按层级条件化 p
    at = {}; mt = {}
    for r in res:
        for c, v in r["at_level"].items():
            at[int(c)] = at.get(int(c), 0) + v
        for c, v in r["maint_level"].items():
            mt[int(c)] = mt.get(int(c), 0) + v
    p_by_level = {c: (mt.get(c, 0) / at[c]) for c in sorted(at) if at[c] >= 30}
    # streak 分布
    all_streaks = [s for r in res for s in r["streaks"]]
    sd = {}
    for s in all_streaks:
        sd[s] = sd.get(s, 0) + 1
    maxs = max(all_streaks) if all_streaks else 0
    mean_streak = sum(all_streaks) / len(all_streaks) if all_streaks else 0.0
    print(f"  p_maintain(整体)={p_maintain:.3f}  L̄={Lbar:.2f}  combo奖励占总分={combo_share*100:.1f}%",
          flush=True)
    print(f"  连击链: n={len(all_streaks)} mean_len={mean_streak:.2f} max={maxs}", flush=True)
    print(f"  p_maintain 按当前层级 (层级:p, n≥30): " +
          " ".join(f"{c}:{p_by_level[c]:.2f}" for c in sorted(p_by_level)), flush=True)
    # 诚实赌局价值：常数-p(乐观) vs level-conditional(实测衰减)。assumed 档 u=combo_unit。
    u_add = combo_unit_value(assumed(), Lbar)
    print(f"  >> 赌局价值对比 (assumed 档 u={u_add:.0f}) 期望未来 combo 奖励 G(C):", flush=True)
    print(f"     {'C':>3} {'G_const(p̄)':>11} {'G_emp(衰减p)':>13} {'高估倍数':>9}", flush=True)
    g_cmp = {}
    for C in (1, 2, 3, 4, 5):
        gc = G(C, p_maintain, u_add)
        ge = G_levelcond(C, p_by_level, u_add)
        g_cmp[C] = {"G_const": gc, "G_emp": ge}
        print(f"     {C:>3} {gc:>11.0f} {ge:>13.0f} {gc/ge if ge>0 else float('nan'):>9.1f}x", flush=True)
    return {"N": N, "T": T, "B": B, "p_maintain": p_maintain, "Lbar": Lbar,
            "combo_bonus_total": combo_bonus, "score_total": total, "combo_share": combo_share,
            "p_by_level": {str(c): p_by_level[c] for c in p_by_level},
            "streak_n": len(all_streaks), "streak_mean": mean_streak, "streak_max": maxs,
            "streak_dist": {str(k): sd[k] for k in sorted(sd)},
            "gamble_value_const_vs_emp": {str(C): g_cmp[C] for C in g_cmp}}


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    mode = sys.argv[1] if len(sys.argv) > 1 else "calc"
    if mode == "calc":
        json.dump(calc_report(), open("combo_gamble_calc.json", "w"), indent=2)
        print("\n-> wrote combo_gamble_calc.json", flush=True)
    elif mode == "empirical":
        N = int(sys.argv[2]) if len(sys.argv) > 2 else 120
        T = int(sys.argv[3]) if len(sys.argv) > 3 else 50
        emp = empirical(N, T)
        # 用实测 p 与 L̄ 驱动计算器，给出"实测点上的豪赌决策"
        rep = calc_report(p_emp=emp["p_maintain"], Lbar=emp["Lbar"])
        json.dump({"empirical": emp, "calculator": rep},
                  open("combo_gamble.json", "w"), indent=2)
        print("\n-> wrote combo_gamble.json", flush=True)
