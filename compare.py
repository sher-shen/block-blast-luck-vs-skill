"""
配对头对头：strong / LA_D3 / LA_D5 跑同一批种子(共享 deal 流)，比较 per-seed 差值。
配对设计消除巨大的 between-seed 方差，少量种子即可判断"更强玩家是否真的更强 + 是否收敛"。
"""
import random
import sys
from fast import NUM_TYPES, strong_hand, make_lookahead


def play(policy, seed, needs_ctx=False, max_rounds=100000):
    deal = random.Random(f"deal-{seed}")
    board = 0; combo = 0; total = 0; move = 0
    for _ in range(max_rounds):
        hand = [deal.randrange(NUM_TYPES) for _ in range(3)]
        if needs_ctx:
            pts, board, combo, alive = policy(board, combo, hand, None, seed=seed, move=move)
        else:
            pts, board, combo, alive = policy(board, combo, hand)
        total += pts; move += 1
        if not alive:
            break
    return total


def mean(xs): return sum(xs) / len(xs)
def std(xs):
    m = mean(xs); return (sum((x-m)**2 for x in xs)/len(xs))**0.5


if __name__ == "__main__":
    K = int(sys.argv[1]) if len(sys.argv) > 1 else 15
    la3 = make_lookahead(D=3, S=8, B=10, base="greedy")
    la_sb = make_lookahead(D=2, S=4, B=8, base="strong")   # strong 基 rollout(便宜参数)

    S_strong, S_la3, S_sb = [], [], []
    for seed in range(K):
        s = play(strong_hand, seed)
        a = play(la3, seed, needs_ctx=True)
        b = play(la_sb, seed, needs_ctx=True)
        S_strong.append(s); S_la3.append(a); S_sb.append(b)
        print(f"seed {seed:3d}: strong={s:6d}  LA_greedy={a:6d}  LA_strongbase={b:6d}")

    print(f"\n{'玩家':>14} {'均值':>8} {'std':>8}")
    for nm, xs in [("strong", S_strong), ("LA_greedy", S_la3), ("LA_strongbase", S_sb)]:
        print(f"{nm:>14} {mean(xs):8.0f} {std(xs):8.0f}")

    def paired(a, b, na, nb):
        d = [x - y for x, y in zip(a, b)]
        se = std(d) / len(d) ** 0.5
        wins = sum(1 for x in d if x > 0)
        print(f"  {na} - {nb}: 配对均差={mean(d):+.0f} ± {se:.0f}(SE)  "
              f"{na}胜 {wins}/{len(d)}")

    print("\n=== 配对差值 (CRN, 同种子) ===")
    paired(S_la3, S_strong, "LA_greedy", "strong")
    paired(S_sb, S_strong, "LA_strongbase", "strong")
    paired(S_sb, S_la3, "LA_strongbase", "LA_greedy")
