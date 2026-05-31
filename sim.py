"""
8x8 Block-Blast 模拟器 + 运气/技能方差分解。
计分用非线性模型 (scoring.py)：multi-clear / combo 连击 / all-clear 清盘奖励。

规则
----
- 8x8 棋盘，每轮发 3 个方块（每个从 CATALOG 等概率独立抽取），全放完才刷新。
- 放下后填满的整行/整列立即消除（不下落）。
- 连击 combo 计数贯穿整局：某次放置触发消除则 +1，未触发归 0。
- 手里还有方块但无合法位置 -> 游戏结束。
- 得分 = scoring.py 的非线性总分。

策略：random / greedy / strong（见各函数注释）。
同种子：三策略吃同一条方块流，双因素方差分解。
"""

import random
from itertools import permutations
from pieces import CATALOG
from scoring import Scoring, score_placement

N = 8
SHAPES = [cells for _, cells in CATALOG]
NUM_TYPES = len(SHAPES)
SCORING = Scoring()
SCORE_W = 1.0   # 策略评估里"真实得分增量"的权重 (相对棋盘健康度启发式)


# ---------- 棋盘基本操作 ----------
def empty_board():
    return [[False] * N for _ in range(N)]


def can_place(board, cells, r0, c0):
    for r, c in cells:
        rr, cc = r0 + r, c0 + c
        if rr < 0 or rr >= N or cc < 0 or cc >= N or board[rr][cc]:
            return False
    return True


def legal_positions(board, cells):
    return [(r0, c0) for r0 in range(N) for c0 in range(N)
            if can_place(board, cells, r0, c0)]


def apply_move(board, cells, r0, c0):
    """原地放置+消除。返回 (消除条数, 放置后是否空盘)。"""
    for r, c in cells:
        board[r0 + r][c0 + c] = True
    full_rows = [r for r in range(N) if all(board[r])]
    full_cols = [c for c in range(N) if all(board[r][c] for r in range(N))]
    for r in full_rows:
        for c in range(N):
            board[r][c] = False
    for c in full_cols:
        for r in range(N):
            board[r][c] = False
    cleared = len(full_rows) + len(full_cols)
    empty = all(not board[r][c] for r in range(N) for c in range(N))
    return cleared, empty


def copy_board(board):
    return [row[:] for row in board]


# ---------- 启发式：棋盘健康度 ----------
def heuristic(board):
    """越大越好。少填格子 + 空格尽量连成片。"""
    filled = 0
    frag = 0
    for r in range(N):
        for c in range(N):
            if board[r][c]:
                filled += 1
            else:
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    rr, cc = r + dr, c + dc
                    if rr < 0 or rr >= N or cc < 0 or cc >= N or board[rr][cc]:
                        frag += 1
    return -1.0 * filled - 0.5 * frag


def best_pos_for_piece(board, cells, combo):
    """给定棋盘+方块+当前连击，返回使 (真实得分*权重 + 健康度) 最大的放法。
    返回 (新棋盘, 本步得分, 新连击, 评估值)；放不下返回 None。"""
    best = None
    ncells = len(cells)
    for (r0, c0) in legal_positions(board, cells):
        b = copy_board(board)
        cleared, empty = apply_move(b, cells, r0, c0)
        pts, ncombo = score_placement(SCORING, ncells, cleared, empty, combo)
        val = pts * SCORE_W + heuristic(b)
        if best is None or val > best[3]:
            best = (b, pts, ncombo, val)
    return best


# ---------- 三种策略：输入棋盘+手牌+连击，返回 (本手得分, 新棋盘, 新连击, 是否存活) ----------
def policy_random(board, hand, combo, rng):
    total = 0
    for cells in hand:
        pos = legal_positions(board, cells)
        if not pos:
            return total, board, combo, False
        r0, c0 = rng.choice(pos)
        cleared, empty = apply_move(board, cells, r0, c0)
        pts, combo = score_placement(SCORING, len(cells), cleared, empty, combo)
        total += pts
    return total, board, combo, True


def policy_greedy(board, hand, combo, rng):
    total = 0
    for cells in hand:
        res = best_pos_for_piece(board, cells, combo)
        if res is None:
            return total, board, combo, False
        board, pts, combo, _ = res
        total += pts
    return total, board, combo, True


def policy_strong(board, hand, combo, rng):
    """枚举 3 个方块的放置顺序，每种贪心放置，取整手真实得分最优。"""
    best_seq = None  # (total_pts, final_board, final_combo)
    for order in set(permutations(range(3))):
        b = copy_board(board)
        c = combo
        total = 0
        ok = True
        for idx in order:
            res = best_pos_for_piece(b, hand[idx], c)
            if res is None:
                ok = False
                break
            b, pts, c, _ = res
            total += pts
        if not ok:
            continue
        val = total * SCORE_W + heuristic(b)
        if best_seq is None or val > best_seq[0]:
            best_seq = (val, total, b, c)
    if best_seq is None:
        return 0, board, combo, False
    _, total, b, c = best_seq
    return total, b, c, True


POLICIES = {"random": policy_random, "greedy": policy_greedy, "strong": policy_strong}


# ---------- 单局 ----------
def play_one_game(policy_fn, seed, max_rounds=100000):
    deal_rng = random.Random(f"deal-{seed}")
    act_rng = random.Random(f"act-{seed}")
    board = empty_board()
    combo = 0
    total_score = 0
    for _ in range(max_rounds):
        hand = [SHAPES[deal_rng.randrange(NUM_TYPES)] for _ in range(3)]
        pts, board, combo, alive = policy_fn(board, hand, combo, act_rng)
        total_score += pts
        if not alive:
            break
    return total_score


# ---------- 方差分解 ----------
def variance_decomposition(num_seeds=300):
    names = list(POLICIES.keys())
    scores = {p: [] for p in names}
    for s in range(num_seeds):
        for p in names:
            scores[p].append(play_one_game(POLICIES[p], s))

    P, S = len(names), num_seeds
    allvals = [scores[p][s] for p in names for s in range(S)]
    grand = sum(allvals) / len(allvals)
    mean_p = {p: sum(scores[p]) / S for p in names}
    mean_s = [sum(scores[p][s] for p in names) / P for s in range(S)]

    ss_total = sum((x - grand) ** 2 for x in allvals)
    ss_policy = S * sum((mean_p[p] - grand) ** 2 for p in names)
    ss_seed = P * sum((m - grand) ** 2 for m in mean_s)
    ss_inter = ss_total - ss_policy - ss_seed
    return names, scores, mean_p, {
        "ss_total": ss_total, "ss_policy": ss_policy,
        "ss_seed": ss_seed, "ss_inter": ss_inter}


def stdev(xs):
    m = sum(xs) / len(xs)
    return (sum((x - m) ** 2 for x in xs) / len(xs)) ** 0.5


def subset_decomp(scores, sub, S):
    allv = [scores[p][s] for p in sub for s in range(S)]
    grand = sum(allv) / len(allv)
    mp = {p: sum(scores[p]) / S for p in sub}
    ms = [sum(scores[p][s] for p in sub) / len(sub) for s in range(S)]
    sst = sum((x - grand) ** 2 for x in allv)
    ssp = S * sum((mp[p] - grand) ** 2 for p in sub)
    sss = len(sub) * sum((m - grand) ** 2 for m in ms)
    return ssp / sst, sss / sst, (sst - ssp - sss) / sst


if __name__ == "__main__":
    import sys
    nseeds = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    print(f"piece types={NUM_TYPES}  seeds={nseeds}  "
          f"scoring: line_base={SCORING.line_base} combo_unit={SCORING.combo_unit} "
          f"all_clear={SCORING.all_clear_bonus}\n")
    names, scores, mean_p, ss = variance_decomposition(nseeds)

    print("=== 各策略平均得分 (非线性计分) ===")
    for p in names:
        xs = scores[p]
        print(f"  {p:7s}: mean={mean_p[p]:9.1f}  std={stdev(xs):8.1f}  "
              f"min={min(xs):5d}  max={max(xs):7d}")

    tot = ss["ss_total"]
    print("\n=== 全体(含 random) 方差分解 ===")
    print(f"  技能 {ss['ss_policy']/tot*100:5.1f}% | 运气 {ss['ss_seed']/tot*100:5.1f}% "
          f"| 交互 {ss['ss_inter']/tot*100:5.1f}%")

    skp, slk, sin = subset_decomp(scores, ["greedy", "strong"], nseeds)
    print("\n=== 只在熟练玩家之间(greedy vs strong) ===")
    print(f"  技能 {skp*100:5.1f}% | 运气 {slk*100:5.1f}% | 交互 {sin*100:5.1f}%")
    print(f"\n  strong 跨 seed 的 CV = {stdev(scores['strong'])/mean_p['strong']*100:.0f}%")
