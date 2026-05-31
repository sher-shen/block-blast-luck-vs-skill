"""
Bitboard 快引擎 + 更强玩家（beam strong / lookahead 前瞻）。
按审核意见实现：
  - 64-bit 整数棋盘，预计算每个 piece 在每个落点的 mask，can_place=按位与。
  - 启发式用位运算 + popcount（无 64 格 Python 循环）。
  - strong = 手内 beam 搜索（联合搜 3 块的顺序×落点，**连击状态贯穿整手**）。
  - lookahead = beam 出 B 个候选末盘 → 每个用 flat Monte Carlo rollout(greedy, D 手)
    估未来分(λ=1，rollout 返回真实分单位) → 选 即时分+平均rollout 最大者。
  - rollout RNG 用 f"rollout-{seed}-{move}-{cand}-{s}"，独立于 deal/act，保 CRN 结构。
"""

import random
from scoring import Scoring, score_placement
from pieces import CATALOG

N = 8
FULL = (1 << 64) - 1
SCORING = Scoring()

# 列/行掩码
COL = [sum(1 << (r * 8 + c) for r in range(8)) for c in range(8)]
ROW = [sum(1 << (r * 8 + c) for c in range(8)) for r in range(8)]
COL0, COL7 = COL[0], COL[7]
ROW0, ROW7 = ROW[0], ROW[7]
NOTC7 = FULL & ~COL7
NOTC0 = FULL & ~COL0
NOTR7 = FULL & ~ROW7
NOTR0 = FULL & ~ROW0

# 每个 piece：单元数 + 所有合法落点的 mask 列表
PIECE_CELLS = [cells for _, cells in CATALOG]
NUM_TYPES = len(PIECE_CELLS)
NCELLS = [len(c) for c in PIECE_CELLS]


def _placements(cells):
    masks = []
    maxr = max(r for r, _ in cells)
    maxc = max(c for _, c in cells)
    for r0 in range(N - maxr):
        for c0 in range(N - maxc):
            m = 0
            for r, c in cells:
                m |= 1 << ((r0 + r) * 8 + (c0 + c))
            masks.append(m)
    return masks


PLACE = [_placements(c) for c in PIECE_CELLS]   # PLACE[piece_id] = [mask,...]


def heuristic(board):
    """越大越好：-已填 - 0.5*碎片度(空格邻接墙/填充的方向数)。位运算实现。"""
    filled = board.bit_count()
    empty = (~board) & FULL
    # 各方向：邻居为 填充 或 墙 -> 对该空格算 1 次碎片
    right = ((board >> 1) & NOTC7) | COL7
    left = ((board << 1) & NOTC0 & FULL) | COL0
    down = ((board >> 8) & NOTR7) | ROW7
    up = ((board << 8) & NOTR0 & FULL) | ROW0
    frag = ((empty & right).bit_count() + (empty & left).bit_count()
            + (empty & down).bit_count() + (empty & up).bit_count())
    return -1.0 * filled - 0.5 * frag


def apply_mask(board, mask):
    """放置(已确认合法)+消除。返回 (新盘, 消除条数, 是否空盘)。"""
    b = board | mask
    clearmask = 0
    cleared = 0
    for r in range(8):
        if b & ROW[r] == ROW[r]:
            clearmask |= ROW[r]
            cleared += 1
    for c in range(8):
        if b & COL[c] == COL[c]:
            clearmask |= COL[c]
            cleared += 1
    b &= ~clearmask
    return b, cleared, (b == 0)


def legal_masks(board, piece_id):
    return [m for m in PLACE[piece_id] if not (board & m)]


# ---------- 策略 ----------
def greedy_hand(board, combo, hand, rng=None):
    """按发牌顺序，每块选 pts+heuristic 最大的落点。"""
    total = 0
    for pid in hand:
        best = None
        for m in PLACE[pid]:
            if board & m:
                continue
            nb, cl, empty = apply_mask(board, m)
            pts, nc = score_placement(SCORING, NCELLS[pid], cl, empty, combo)
            val = pts + heuristic(nb)
            if best is None or val > best[0]:
                best = (val, nb, nc, pts)
        if best is None:
            return total, board, combo, False
        _, board, combo, pts = best
        total += pts
    return total, board, combo, True


def beam_hand(board, combo, hand, B=12):
    """手内 beam：联合搜 3 块顺序×落点，连击贯穿。返回 top-B 末态 [(board,combo,score)]。"""
    # state: (board, combo, score, used_tuple)
    states = [(board, combo, 0, ())]
    for _ in range(3):
        cand = []
        for (b, c, sc, used) in states:
            for i in range(3):
                if i in used:
                    continue
                pid = hand[i]
                for m in PLACE[pid]:
                    if b & m:
                        continue
                    nb, cl, empty = apply_mask(b, m)
                    pts, nc = score_placement(SCORING, NCELLS[pid], cl, empty, c)
                    cand.append((nb, nc, sc + pts, used + (i,)))
        if not cand:
            return []   # 这一步谁都放不下 -> 死
        cand.sort(key=lambda s: s[2] + heuristic(s[0]), reverse=True)
        states = cand[:B]
    return [(b, c, sc) for (b, c, sc, _) in states]


def strong_hand(board, combo, hand, rng=None, B=12):
    res = beam_hand(board, combo, hand, B)
    if not res:
        return 0, board, combo, False
    best = max(res, key=lambda s: s[2] + heuristic(s[0]))
    return best[2], best[0], best[1], True


def _rollout(board, combo, depth, rng):
    """从 (board,combo) 用 greedy 滚 depth 手随机牌，返回累计分。"""
    total = 0
    for _ in range(depth):
        hand = [rng.randrange(NUM_TYPES) for _ in range(3)]
        pts, board, combo, alive = greedy_hand(board, combo, hand)
        total += pts
        if not alive:
            break
    return total


def _rollout_fixed_local(board, combo, future_pieces):
    """按给定牌流用 greedy 滚动，返回累计分(无随机，供 CRN 配对)。"""
    total = 0
    for i in range(0, len(future_pieces), 3):
        for pid in future_pieces[i:i + 3]:
            best = None
            for m in PLACE[pid]:
                if board & m:
                    continue
                nb, cl, empty = apply_mask(board, m)
                pts, ncb = score_placement(SCORING, NCELLS[pid], cl, empty, combo)
                val = pts + heuristic(nb)
                if best is None or val > best[0]:
                    best = (val, nb, ncb, pts)
            if best is None:
                return total
            _, board, combo, pts = best
            total += pts
    return total


def _rollout_fixed_strong(board, combo, future_pieces, B=8):
    """按给定牌流用 beam-strong 滚动(更强基策略)。"""
    total = 0
    for i in range(0, len(future_pieces), 3):
        hand = future_pieces[i:i + 3]
        res = beam_hand(board, combo, hand, B)
        if not res:
            return total
        best = max(res, key=lambda s: s[2] + heuristic(s[0]))
        total += best[2]; board = best[0]; combo = best[1]
    return total


def make_lookahead(D=3, S=10, B=12, base="greedy"):
    """beam 出 B 候选 -> 各 rollout S 次 D 手 -> 选最优。base=rollout 基策略。"""
    roll_fn = _rollout_fixed_local if base == "greedy" else \
        (lambda nb, nc, fs: _rollout_fixed_strong(nb, nc, fs, B=6))

    def policy(board, combo, hand, rng, seed=0, move=0):
        cands = beam_hand(board, combo, hand, B)
        if not cands:
            return 0, board, combo, False
        future_streams = [
            [random.Random(f"rollout-{seed}-{move}-{s}").randrange(NUM_TYPES)
             for _ in range(3 * D)] for s in range(S)]
        best = None
        for (nb, nc, hand_score) in cands:
            roll = sum(roll_fn(nb, nc, fs) for fs in future_streams)
            value = hand_score + roll / S + heuristic(nb)
            if best is None or value > best[0]:
                best = (value, hand_score, nb, nc)
        return best[1], best[2], best[3], True
    return policy
