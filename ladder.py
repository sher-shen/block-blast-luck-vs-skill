"""
技能阶梯：用 ε-greedy 混合(以概率 ε 整手乱放，否则用 strong)造出从"纯新手"到"接近最优"
的连续技能轴。对每一级测：平均分 μ、运气波动 σ(跨 seed)、以及"再升一级技能"的边际收益。

回答："多少分以下是技能主导、多少分以上是运气主导？"
判据：技能缺口 (到天花板还差多少) vs 运气波动 σ。
  缺口 > σ  -> 技能主导（你分低主要因为策略差，练能涨）
  缺口 < σ  -> 运气主导（你已接近天花板，单局高低主要看牌运）
交叉点 = 缺口 ≈ σ 处的分数，就是"新手↔高手"的真正分界。
"""

import random
from sim import (SHAPES, NUM_TYPES, empty_board, policy_random, policy_strong,
                 stdev)


def policy_eps(board, hand, combo, rng, eps):
    if rng.random() < eps:
        return policy_random(board, hand, combo, rng)
    return policy_strong(board, hand, combo, rng)


def play_eps(seed, eps, max_rounds=100000):
    deal_rng = random.Random(f"deal-{seed}")
    act_rng = random.Random(f"act-{seed}")
    board = empty_board()
    combo = 0
    total = 0
    for _ in range(max_rounds):
        hand = [SHAPES[deal_rng.randrange(NUM_TYPES)] for _ in range(3)]
        pts, board, combo, alive = policy_eps(board, hand, combo, act_rng, eps)
        total += pts
        if not alive:
            break
    return total


def percentile(xs, p):
    s = sorted(xs)
    k = (len(s) - 1) * p / 100
    lo = int(k)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


if __name__ == "__main__":
    import sys
    M = int(sys.argv[1]) if len(sys.argv) > 1 else 150
    eps_list = [1.0, 0.95, 0.9, 0.8, 0.6, 0.4, 0.2, 0.1, 0.0]

    rungs = []  # (eps, mean, std, p10, p50, p90)
    for eps in eps_list:
        scores = [play_eps(s, eps) for s in range(M)]
        m = sum(scores) / len(scores)
        rungs.append((eps, m, stdev(scores),
                      percentile(scores, 10), percentile(scores, 50),
                      percentile(scores, 90)))

    ceiling = max(r[1] for r in rungs)  # 最强一级的平均分 = 技能天花板
    print(f"M={M} seeds/级。技能天花板(strong 平均) = {ceiling:.0f}\n")
    print(f"{'技能(1-ε)':>9} {'平均μ':>9} {'运气σ':>9} {'p10':>7} {'p50':>7} "
          f"{'p90':>8} {'到顶缺口':>9} {'缺口/σ':>7} {'谁主导':>8}")
    for (eps, m, sd, p10, p50, p90) in rungs:
        gap = ceiling - m
        ratio = gap / sd if sd > 0 else 0
        who = "技能" if gap > sd else "运气"
        print(f"{1-eps:9.2f} {m:9.0f} {sd:9.0f} {p10:7.0f} {p50:7.0f} "
              f"{p90:8.0f} {gap:9.0f} {ratio:7.2f} {who:>8}")

    # 找交叉点：缺口从 >σ 变到 <σ 的那一级，线性插值出分数
    cross = None
    for i in range(len(rungs) - 1):
        m1, sd1 = rungs[i][1], rungs[i][2]
        m2, sd2 = rungs[i + 1][1], rungs[i + 1][2]
        d1, d2 = ceiling - m1 - sd1, ceiling - m2 - sd2  # 缺口-σ
        if d1 > 0 >= d2:
            t = d1 / (d1 - d2)
            cross = m1 + t * (m2 - m1)
            break
    print("\n" + "=" * 60)
    if cross:
        print(f"交叉点(缺口≈σ)的平均分 ≈ {cross:.0f}")
        print(f"  → 长期平均分 < ~{cross:.0f}：技能主导（练策略能稳涨）")
        print(f"  → 长期平均分 > ~{cross:.0f}：运气主导（接近天花板，单局看牌运）")
    else:
        print("未找到明确交叉点（可能全程某一方主导）")
