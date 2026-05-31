"""
计分模型 (可调)。Block Blast 类游戏的真实得分是**非线性**的：
  1) 单次放置同时消除多行/列 -> 越多越值钱 (multi-clear 奖励)
  2) 连续多次放置都触发消除 -> 连击 combo 奖励 (streak)
  3) 把棋盘清空 -> all-clear 大奖励

下面是一套**默认数值（假设，非游戏实测）**，全部集中在此便于调。
"""

from dataclasses import dataclass


@dataclass
class Scoring:
    per_cell: int = 1            # 每放下 1 格 +1（放置分，量级小）
    line_base: int = 10          # 消除的基础单位

    # 单次放置消除 L 条(行+列合计) 的得分：三角数增长，越多越超值
    #   L=1 ->10, L=2 ->30, L=3 ->60, L=4 ->100, L=5 ->150 ...
    def clear_points(self, L: int) -> int:
        return self.line_base * L * (L + 1) // 2

    # 连击：本次放置触发消除，且处于连续消除链中。streak=本次是链上第几次(从1起)
    #   streak=1 不给(第一次不算连击)，2 ->+50, 3 ->+100 ...
    combo_unit: int = 50

    def combo_points(self, streak: int) -> int:
        return self.combo_unit * max(0, streak - 1)

    # all-clear：放置后整盘为空
    all_clear_bonus: int = 300


def score_placement(scoring: Scoring, num_cells: int, lines_cleared: int,
                    board_empty: bool, combo_before: int):
    """
    返回 (本次得分, 新的连击计数)。
    combo 计数：触发消除则 +1，未触发则归 0。
    """
    pts = scoring.per_cell * num_cells
    if lines_cleared > 0:
        combo_after = combo_before + 1
        pts += scoring.clear_points(lines_cleared)
        pts += scoring.combo_points(combo_after)
        if board_empty:
            pts += scoring.all_clear_bonus
    else:
        combo_after = 0
    return pts, combo_after
