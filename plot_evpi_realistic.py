"""零依赖 SVG：读 evpi_realistic.json，画 A' 的核心结论——
EVPI(运气)占比在 2x2 {assumed,real_approx} x {uniform,benevolent} benchmark 设定下怎么移动。
同 plot_evpi_vla.py 风格。"""
import json
import os

R = json.load(open("evpi_realistic.json", encoding="utf-8"))
os.makedirs("figures", exist_ok=True)
TS = R["Ts"]
CONDS = [("assumed_uniform", "#9bb7d4", "assumed · uniform (原 benchmark)"),
         ("assumed_benevolent", "#6fae8f", "assumed · benevolent(去3x3)"),
         ("real_approx_uniform", "#d9a55b", "real_approx · uniform"),
         ("real_approx_benevolent", "#c8623f", "real_approx · benevolent (最真实)")]

W, H = 960, 560
out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
       f'font-family="Helvetica,Arial,sans-serif" font-size="13">',
       f'<rect width="{W}" height="{H}" fill="white"/>']
out.append(f'<text x="{W/2}" y="26" font-size="17" text-anchor="middle" font-weight="bold">'
           f'A&#39;: 运气(EVPI)占比对 benchmark 定义的敏感性 — 真实化只把占比推得更高</text>')
out.append(f'<text x="{W/2}" y="46" font-size="11" text-anchor="middle" fill="#555">'
           f'N={R["N"]} · D_oracle={R["D_oracle"]} · strong-分母三玩家 EVPI=(seer-blind)/(seer-strong) '
           f'· 固定 T · per-seed 配对中位数 + bootstrap 95% CI</text>')

# ---- 左：分组柱，每 T 四个 condition ----
ML, MT = 56, 84
PW, PH = 540, 380
x0 = ML
out.append(f'<text x="{x0}" y="{MT-10}" font-weight="bold">① EVPI 占比 (越高=缺口里运气越主导)</text>')


def y_share(v):     # 0..1.30
    return MT + PH * (1 - v / 1.30)


for v in (0, 0.25, 0.5, 0.75, 1.0, 1.25):
    yy = y_share(v)
    out.append(f'<line x1="{x0}" y1="{yy}" x2="{x0+PW}" y2="{yy}" stroke="#eee"/>')
    out.append(f'<text x="{x0-6}" y="{yy+4}" text-anchor="end" font-size="11">{int(v*100)}%</text>')
out.append(f'<line x1="{x0}" y1="{y_share(1.0)}" x2="{x0+PW}" y2="{y_share(1.0)}" '
           f'stroke="#d33" stroke-width="1.3" stroke-dasharray="4,3"/>')
out.append(f'<text x="{x0+PW}" y="{y_share(1.0)-4}" text-anchor="end" font-size="10" fill="#d33">'
           f'100% = 残差全是信息价值(运气)</text>')

gw = PW / len(TS)
bw = gw / (len(CONDS) + 1)
for ti, T in enumerate(TS):
    gx = x0 + gw * ti
    for ci, (cond, col, _lab) in enumerate(CONDS):
        d = R["conditions"][cond][f"T{T}"]
        sh = d["EVPI_share_strong_denom"]
        v = sh["median"]
        if v != v:       # nan
            continue
        bx = gx + bw * (ci + 0.5)
        out.append(f'<rect x="{bx}" y="{y_share(v)}" width="{bw*0.86}" '
                   f'height="{MT+PH-y_share(v)}" fill="{col}"/>')
        out.append(f'<text x="{bx+bw*0.43}" y="{y_share(v)-3}" text-anchor="middle" '
                   f'font-size="10" font-weight="bold" fill="#333">{v*100:.0f}</text>')
        ci_lo, ci_hi = sh["ci"]
        if ci_lo == ci_lo:
            out.append(f'<line x1="{bx+bw*0.43}" y1="{y_share(ci_lo)}" x2="{bx+bw*0.43}" '
                       f'y2="{y_share(ci_hi)}" stroke="#555" stroke-width="1"/>')
    out.append(f'<text x="{gx+gw/2}" y="{MT+PH+18}" text-anchor="middle" font-weight="bold">T={T}</text>')
    out.append(f'<text x="{gx+gw/2}" y="{MT+PH+33}" text-anchor="middle" font-size="10" fill="#888">'
               f'cohort {R["conditions"]["assumed_uniform"][f"T{T}"]["cohort_n"]}'
               f'→{R["conditions"]["real_approx_benevolent"][f"T{T}"]["cohort_n"]}</text>')
# 图例
ly = MT + PH + 50
for i, (cond, col, lab) in enumerate(CONDS):
    yy = ly + (i // 2) * 18
    xx = x0 + (i % 2) * 270
    out.append(f'<rect x="{xx}" y="{yy-10}" width="12" height="12" fill="{col}"/>')
    out.append(f'<text x="{xx+17}" y="{yy}" font-size="11">{lab}</text>')

# ---- 右：分数阶梯坍缩（real_approx 下 strong→seer 缺口收窄） ----
RX = 660
RPW, RPH = 250, 380
out.append(f'<text x="{RX}" y="{MT-10}" font-weight="bold">② 分数阶梯 (T=50 cohort 中位)</text>')
players = [("strong", "#9bb7d4"), ("blind", "#b9a0d4"), ("seer", "#5aa469")]
panels = [("assumed_uniform", "assumed·uni"), ("real_approx_benevolent", "real·benev")]
allvals = []
for cond, _ in panels:
    dd = R["conditions"][cond]["T50"]["score_median_cohort"]
    allvals += [dd[p] for p, _ in players]
vmax = max(allvals) * 1.14
sub_w = RPW / len(panels)


def y_sc(v):
    return MT + RPH * (1 - v / vmax)


for v in range(0, int(vmax) + 1, 1000):
    yy = y_sc(v)
    out.append(f'<line x1="{RX}" y1="{yy}" x2="{RX+RPW}" y2="{yy}" stroke="#eee"/>')
    out.append(f'<text x="{RX-4}" y="{yy+4}" text-anchor="end" font-size="10">{v}</text>')
for pi, (cond, plab) in enumerate(panels):
    dd = R["conditions"][cond]["T50"]["score_median_cohort"]
    px = RX + sub_w * pi
    bw3 = sub_w / (len(players) + 1)
    for i, (p, col) in enumerate(players):
        v = dd[p]
        bx = px + bw3 * (i + 0.5)
        out.append(f'<rect x="{bx}" y="{y_sc(v)}" width="{bw3*0.8}" height="{MT+RPH-y_sc(v)}" fill="{col}"/>')
        out.append(f'<text x="{bx+bw3*0.4}" y="{y_sc(v)-3}" text-anchor="middle" font-size="9">{v:.0f}</text>')
    out.append(f'<text x="{px+sub_w/2}" y="{MT+RPH+16}" text-anchor="middle" font-size="10">{plab}</text>')
out.append(f'<text x="{RX}" y="{MT+RPH+40}" font-size="10" fill="#555">strong(蓝)/blind(紫)/seer(绿)</text>')
out.append(f'<text x="{RX}" y="{MT+RPH+56}" font-size="10" fill="#c8623f">real_approx 下绝对缺口收窄,</text>')
out.append(f'<text x="{RX}" y="{MT+RPH+70}" font-size="10" fill="#c8623f">但其中运气占比反升</text>')

out.append(f'<text x="{ML}" y="{H-10}" font-size="11" fill="#444">'
           f'结论：57–69%(原 assumed·uniform)是 benchmark 压低的**保守下界**；越贴近真实游戏'
           f'(乘法连击计分 + 善意发牌)，运气占比越逼近 ~100% ⇒ 天花板处技能通道被吃满的结论更强。</text>')
out.append('</svg>')
open("figures/evpi_realistic.svg", "w", encoding="utf-8").write("\n".join(out))
print("-> figures/evpi_realistic.svg")
