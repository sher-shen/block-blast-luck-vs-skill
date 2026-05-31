"""
方块目录 (piece catalog) for 8x8 Block-Blast-style game.

设计原则
--------
- 用户假设：每一种"不同的格子(方块)"出现概率相等。
- 我们把"不同朝向"也算作不同的方块（因为游戏里横条和竖条是分别发的，
  斜的两种对角线也分别发）。所以下面对每个基础形状生成它所有不重复的旋转，
  每个朝向 = 一个独立的、等概率的 piece type。
- 覆盖用户描述的范围：1 格 / 2 格(横,竖,两条对角线) / 3 / 4 / 5(最长横竖)
  / 一直到 9 格(3x3)。

每个 piece 用一组 (row, col) 单元格坐标表示，左上角已归一化到 (0,0)。
"""

from itertools import product


def normalize(cells):
    """把坐标平移到左上角对齐 (min row=0, min col=0)，并排序去重。"""
    mr = min(r for r, _ in cells)
    mc = min(c for _, c in cells)
    return tuple(sorted((r - mr, c - mc) for r, c in cells))


def rotate90(cells):
    """顺时针旋转 90 度: (r,c) -> (c, -r)。"""
    return normalize([(c, -r) for r, c in cells])


def all_orientations(cells):
    """返回一个基础形状的所有不重复旋转朝向。"""
    seen = set()
    cur = normalize(cells)
    out = []
    for _ in range(4):
        if cur not in seen:
            seen.add(cur)
            out.append(cur)
        cur = rotate90(cur)
    return out


# ---- 基础形状（旋转前的代表）。注释 = 中文名 ----
BASE_SHAPES = {
    # 1 格
    "monomino":      [(0, 0)],

    # 2 格
    "domino":        [(0, 0), (0, 1)],            # 横/竖 由旋转生成
    "diag2":         [(0, 0), (1, 1)],            # 斜两格 (主对角)
    "antidiag2":     [(0, 1), (1, 0)],            # 斜两格 (反对角)

    # 3 格
    "tromino_I":     [(0, 0), (0, 1), (0, 2)],    # 一字三 (横/竖)
    "tromino_L":     [(0, 0), (1, 0), (1, 1)],    # L 形三 (4 朝向)
    "diag3":         [(0, 0), (1, 1), (2, 2)],    # 斜三格 (主)
    "antidiag3":     [(0, 2), (1, 1), (2, 0)],    # 斜三格 (反)

    # 4 格
    "tetro_I":       [(0, 0), (0, 1), (0, 2), (0, 3)],   # 一字四
    "tetro_O":       [(0, 0), (0, 1), (1, 0), (1, 1)],   # 2x2 方块
    "tetro_T":       [(0, 0), (0, 1), (0, 2), (1, 1)],   # T
    "tetro_L":       [(0, 0), (1, 0), (2, 0), (2, 1)],   # L
    "tetro_J":       [(0, 1), (1, 1), (2, 1), (2, 0)],   # J
    "tetro_S":       [(0, 1), (0, 2), (1, 0), (1, 1)],   # S
    "tetro_Z":       [(0, 0), (0, 1), (1, 1), (1, 2)],   # Z

    # 5 格
    "pent_I":        [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],  # 一字五 (最长横/竖)
    "pent_plus":     [(0, 1), (1, 0), (1, 1), (1, 2), (2, 1)],  # 十字

    # 6 格
    "rect_2x3":      [(r, c) for r, c in product(range(2), range(3))],  # 2x3 矩形

    # 9 格
    "square_3x3":    [(r, c) for r, c in product(range(3), range(3))],  # 3x3 大方块
}


def build_catalog():
    """生成完整目录：每个基础形状 -> 所有不重复朝向，去重后每个朝向是一个 piece。"""
    catalog = []          # list of (name, cells)
    seen = set()
    for name, cells in BASE_SHAPES.items():
        for i, ori in enumerate(all_orientations(cells)):
            if ori in seen:
                continue
            seen.add(ori)
            label = name if len(all_orientations(cells)) == 1 else f"{name}#{i}"
            catalog.append((label, ori))
    return catalog


CATALOG = build_catalog()


def describe():
    """打印目录概览。"""
    by_size = {}
    for name, cells in CATALOG:
        by_size.setdefault(len(cells), []).append(name)
    print(f"总 piece 数 (含朝向): {len(CATALOG)}")
    for size in sorted(by_size):
        names = by_size[size]
        print(f"  {size} 格: {len(names):2d} 种 -> {', '.join(names)}")


if __name__ == "__main__":
    describe()
