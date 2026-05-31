"""零依赖 SVG 绘图（纯标准库）：读 results.json，产出两张图到 figures/。
 fig1: 玩家均分对数条形图(random/greedy/strong/oracle) + 技能地板/运气缺口标注。
 fig2: 技能阶梯 缺口/σ 曲线 + 交叉点。
"""
import json
import math
import os

R = json.load(open("results.json"))
os.makedirs("figures", exist_ok=True)


def svg_open(w, h):
    return [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" '
            f'font-family="Helvetica,Arial,sans-serif" font-size="13">',
            f'<rect width="{w}" height="{h}" fill="white"/>']


def text(x, y, s, size=13, anchor="start", color="#222", weight="normal"):
    return (f'<text x="{x}" y="{y}" font-size="{size}" text-anchor="{anchor}" '
            f'fill="{color}" font-weight="{weight}">{s}</text>')


# ---------- fig1: 对数条形图 ----------
def fig1():
    W, H = 760, 380
    s = svg_open(W, H)
    x0, x1 = 150, 700
    players = [("random (乱放)", R["players"]["random"]["mean"], "#bbb"),
               ("greedy (贪心)", R["players"]["greedy"]["mean"], "#7aa7d8"),
               ("strong (在线最强)", R["paired"]["strong"]["mean"], "#2f6db5"),
               ("ORACLE (开未来牌)", R["oracle"]["mean"], "#e07b39")]
    lo, hi = 1, max(p[1] for p in players) * 1.3
    def X(v): return x0 + (math.log10(max(v, 1)) - math.log10(lo)) / \
        (math.log10(hi) - math.log10(lo)) * (x1 - x0)
    s.append(text(W/2, 28, "玩家均分（对数轴）：技能地板 vs 运气缺口", 16, "middle", "#111", "bold"))
    # 网格
    for gv in [1, 10, 100, 1000, 10000]:
        gx = X(gv)
        s.append(f'<line x1="{gx}" y1="60" x2="{gx}" y2="320" stroke="#eee"/>')
        s.append(text(gx, 338, str(gv), 11, "middle", "#999"))
    y = 75; bh = 42
    for (nm, val, col) in players:
        s.append(f'<rect x="{x0}" y="{y}" width="{X(val)-x0:.1f}" height="{bh-12}" fill="{col}" rx="3"/>')
        s.append(text(x0 - 8, y + bh/2 - 2, nm, 13, "end", "#333"))
        s.append(text(X(val) + 6, y + bh/2 - 2, f"{val:.0f}", 12, "start", "#333", "bold"))
        y += bh
    # 标注
    xr, xg, xs2, xo = (X(players[0][1]), X(players[1][1]), X(players[2][1]), X(players[3][1]))
    s.append(text(x0, 305, f"技能地板：乱放→会玩 ≈ {players[2][1]/players[0][1]:.0f}×", 12, "start", "#2f6db5", "bold"))
    s.append(text(x1, 305, f"运气=信息缺口 ≈ {R['oracle_gap_ratio']*100:.0f}%", 12, "end", "#e07b39", "bold"))
    s.append('</svg>')
    open("figures/fig1_players.svg", "w").write("\n".join(s))


# ---------- fig2: 技能阶梯 缺口/σ ----------
def fig2():
    W, H = 760, 400
    s = svg_open(W, H)
    x0, x1, y0, y1 = 90, 700, 70, 320
    lad = R["ladder"]
    means = [r["mean"] for r in lad]
    lo, hi = 1, max(means) * 1.3
    def X(i): return x0 + i / (len(lad) - 1) * (x1 - x0)
    def Y(v): return y1 - (math.log10(max(v, 1)) - math.log10(lo)) / \
        (math.log10(hi) - math.log10(lo)) * (y1 - y0)
    s.append(text(W/2, 28, "技能阶梯：均分 ± 运气波动σ，与天花板", 16, "middle", "#111", "bold"))
    s.append(text(W/2, 48, "缺口(到天花板) > σ → 技能主导； < σ → 运气主导", 12, "middle", "#777"))
    for gv in [1, 10, 100, 1000, 10000]:
        gy = Y(gv)
        s.append(f'<line x1="{x0}" y1="{gy}" x2="{x1}" y2="{gy}" stroke="#eee"/>')
        s.append(text(x0 - 6, gy + 4, str(gv), 11, "end", "#999"))
    # 天花板线
    ceil = R["ceiling"]
    s.append(f'<line x1="{x0}" y1="{Y(ceil)}" x2="{x1}" y2="{Y(ceil)}" stroke="#e07b39" stroke-dasharray="5,4"/>')
    s.append(text(x1, Y(ceil) - 6, f"天花板 {ceil:.0f}", 11, "end", "#e07b39"))
    # 误差带 + 均线
    pts = []
    for i, r in enumerate(lad):
        x = X(i); m = r["mean"]; sd = r["std"]
        s.append(f'<line x1="{x}" y1="{Y(max(m-sd,1)):.1f}" x2="{x}" y2="{Y(m+sd):.1f}" stroke="#9bbce0" stroke-width="6" opacity="0.6"/>')
        pts.append(f"{x:.1f},{Y(m):.1f}")
        col = "#2f6db5" if r["dominant"] == "skill" else "#c0392b"
        s.append(f'<circle cx="{x:.1f}" cy="{Y(m):.1f}" r="4" fill="{col}"/>')
        s.append(text(x, y1 + 18, f"{r['skill_axis']:.1f}", 11, "middle", "#777"))
    s.append(f'<polyline points="{" ".join(pts)}" fill="none" stroke="#2f6db5" stroke-width="2"/>')
    s.append(text(W/2, y1 + 40, "技能水平 (1-ε)", 12, "middle", "#555"))
    if R["crossover"]:
        s.append(text(x0 + 10, y0 + 10, f"交叉点(缺口≈σ) ≈ {R['crossover']:.0f}", 13, "start", "#c0392b", "bold"))
    s.append(text(x0 + 10, y0 + 28, "蓝=技能主导  红=运气主导", 11, "start", "#777"))
    s.append('</svg>')
    open("figures/fig2_ladder.svg", "w").write("\n".join(s))


fig1(); fig2()
print("写出 figures/fig1_players.svg, figures/fig2_ladder.svg")
