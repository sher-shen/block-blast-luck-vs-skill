"""零依赖 SVG：读 endless_survival.json，画生存分布 + T=50 截断标注。
跟 plots.py 同风格(纯标准库 SVG -> figures/)。"""
import json
import os

R = json.load(open("endless_survival.json", encoding="utf-8"))
os.makedirs("figures", exist_ok=True)
survs = R["survs"]
M, T = R["M"], R["T"]

# 直方图(bin=10)，画到 260 截止(>260 极少，单独标注)
BIN = 10
TOP = 260
bins = {}
over = 0
for s in survs:
    if s >= TOP:
        over += 1
    else:
        bins[(s // BIN) * BIN] = bins.get((s // BIN) * BIN, 0) + 1
maxc = max(bins.values())

W, H = 860, 460
ML, MR, MT, MB = 60, 20, 70, 60
PW, PH = W - ML - MR, H - MT - MB


def x_of(v):  # surv 0..TOP -> px
    return ML + PW * v / TOP


def y_of(c):  # count 0..maxc -> px
    return MT + PH * (1 - c / maxc)


out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
       f'font-family="Helvetica,Arial,sans-serif" font-size="13">',
       f'<rect width="{W}" height="{H}" fill="white"/>']
out.append(f'<text x="{W/2}" y="26" font-size="17" text-anchor="middle" font-weight="bold">'
           f'Step A: 无限局(T={T}) 生存回合分布 — 最强策略 D2 S30 base=heur (M={M})</text>')

# T=50 截断带：surv>=50 在旧 benchmark 里被砍到 50
cens = sum(1 for s in survs if s >= 50)
out.append(f'<rect x="{x_of(50)}" y="{MT}" width="{x_of(TOP)-x_of(50)}" height="{PH}" '
           f'fill="#ffecec"/>')
out.append(f'<line x1="{x_of(50)}" y1="{MT}" x2="{x_of(50)}" y2="{MT+PH}" '
           f'stroke="#d33" stroke-width="2" stroke-dasharray="5,4"/>')
out.append(f'<text x="{x_of(50)+6}" y="{MT+16}" fill="#d33" font-weight="bold">'
           f'旧 T=50 封顶</text>')
out.append(f'<text x="{x_of(50)+6}" y="{MT+34}" fill="#d33" font-size="12">'
           f'红区={cens}/{M}={100*cens/M:.0f}% 被截断</text>')

# 均值竖线
nat = R["surv_mean"]
out.append(f'<line x1="{x_of(nat)}" y1="{MT}" x2="{x_of(nat)}" y2="{MT+PH}" '
           f'stroke="#2a7" stroke-width="2"/>')
out.append(f'<text x="{x_of(nat)+5}" y="{MT+PH-8}" fill="#197" font-weight="bold">'
           f'自然 surv 均值={nat:.0f}</text>')
out.append(f'<line x1="{x_of(42.3)}" y1="{MT}" x2="{x_of(42.3)}" y2="{MT+PH}" '
           f'stroke="#999" stroke-width="1.5" stroke-dasharray="3,3"/>')
out.append(f'<text x="{x_of(42.3)-4}" y="{MT+PH-26}" fill="#777" font-size="11" '
           f'text-anchor="end">旧"surv≈42"=截断均值</text>')

# 柱
for lo, c in sorted(bins.items()):
    x = x_of(lo)
    w = x_of(lo + BIN) - x - 1
    y = y_of(c)
    col = "#d88" if lo >= 50 else "#69c"
    out.append(f'<rect x="{x}" y="{y}" width="{w}" height="{MT+PH-y}" fill="{col}"/>')

# 轴
out.append(f'<line x1="{ML}" y1="{MT+PH}" x2="{ML+PW}" y2="{MT+PH}" stroke="#333"/>')
out.append(f'<line x1="{ML}" y1="{MT}" x2="{ML}" y2="{MT+PH}" stroke="#333"/>')
for v in range(0, TOP + 1, 20):
    out.append(f'<text x="{x_of(v)}" y="{MT+PH+18}" text-anchor="middle" font-size="11">{v}</text>')
out.append(f'<text x="{x_of(TOP)+0}" y="{MT+PH+36}" text-anchor="end" font-size="11" fill="#555">'
           f'(另有 {over} 局 surv&gt;{TOP}, 最高 {R["surv_max"]})</text>')
out.append(f'<text x="{ML+PW/2}" y="{H-8}" text-anchor="middle">生存回合 surv (自然死)</text>')
out.append(f'<text x="16" y="{MT+PH/2}" text-anchor="middle" '
           f'transform="rotate(-90 16 {MT+PH/2})">局数</text>')
for c in range(0, maxc + 1, 5):
    out.append(f'<text x="{ML-6}" y="{y_of(c)+4}" text-anchor="end" font-size="11">{c}</text>')

# 结论框
out.append(f'<text x="{ML}" y="{MT-14}" font-size="12" fill="#333">'
           f'总分 {R["total_mean"]:.0f}±{R["total_se"]:.0f} (旧 T=50≈2950) · '
           f'撞 T={T} 封顶 {R["censored_at_T"]}/{M} ⇒ T={T} 基本非 binding</text>')
out.append('</svg>')
open("figures/endless_survival.svg", "w", encoding="utf-8").write("\n".join(out))
print("-> figures/endless_survival.svg")
