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

    # 连击：本次放置触发消除，且处于连续消除链中。streak=本次是链上第几次(从1起)
    #   streak=1 不给(第一次不算连击)，2 ->+50, 3 ->+100 ...
    combo_unit: int = 50

    # all-clear：放置后整盘为空
    all_clear_bonus: int = 300

    # --- 计分档参数化（2026-06-06 Step C 校准）。默认值 = 历史"假设档"，行为逐字不变 ---
    # clear_table 非空时覆盖三角数公式：clear_table[min(L,len)-1]（1-based L）。
    clear_table: tuple = ()
    # combo_mode: "additive"(假设档, combo_points 加法) | "mult"(真实近似档, 乘 clear bonus)
    combo_mode: str = "additive"
    combo_mult: float = 0.5      # mult 档每级连击的倍率增量

    # 单次放置消除 L 条(行+列合计) 的得分：三角数增长，越多越超值
    #   L=1 ->10, L=2 ->30, L=3 ->60, L=4 ->100, L=5 ->150 ...
    def clear_points(self, L: int) -> int:
        if self.clear_table:
            return self.clear_table[min(L, len(self.clear_table)) - 1]
        return self.line_base * L * (L + 1) // 2

    def combo_points(self, streak: int) -> int:
        return self.combo_unit * max(0, streak - 1)


def assumed() -> "Scoring":
    """历史"假设档"（非游戏实测）= 既有默认值。"""
    return Scoring()


def real_approx() -> "Scoring":
    """真实 App "近似档"（社区逆向，非官方，各源不一致 → 仅近似/对外可读用）。
    来源: blockpuzzlesolver.com/scoring, onlineblockblastsolver.com/block-blast-score-rules,
    blocksolver.bitbucket.io。共识: per_cell=1；同手消 L 条的 bonus 表 {1:10,2:20,3:60,4:120,
    5:200,6+:300}；board-clear=360；连击=乘法放大 clear bonus (Score≈Base×(1+combo×mult))。
    连击指数口径各源不一(本档取 1+(streak-1)*0.5 → 首消×1.0 不虚高，保守)。"""
    return Scoring(per_cell=1, all_clear_bonus=360,
                   clear_table=(10, 20, 60, 120, 200, 300),
                   combo_mode="mult", combo_mult=0.5)


def score_placement(scoring: Scoring, num_cells: int, lines_cleared: int,
                    board_empty: bool, combo_before: int):
    """
    返回 (本次得分, 新的连击计数)。
    combo 计数：触发消除则 +1，未触发则归 0。
    """
    pts = scoring.per_cell * num_cells
    if lines_cleared > 0:
        combo_after = combo_before + 1
        if scoring.combo_mode == "mult":
            # 真实近似档：连击乘法放大 clear bonus（首消 streak=1 -> ×1.0）
            pts += scoring.clear_points(lines_cleared) * (
                1.0 + (combo_after - 1) * scoring.combo_mult)
        else:
            pts += scoring.clear_points(lines_cleared)
            pts += scoring.combo_points(combo_after)
        if board_empty:
            pts += scoring.all_clear_bonus
    else:
        combo_after = 0
    return pts, combo_after
