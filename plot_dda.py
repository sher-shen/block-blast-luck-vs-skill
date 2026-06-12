"""零依赖 SVG：读 dda_forward.json，画 Direction B 的"DDA 设计曲线"——
luck-share(EVPI 占比) 与 生存率 如何随发牌难度旋钮 β 变化。
x 轴 = E[每块格数]（β 的可读难度标量，左=善意/易，右=对抗/难）。"""
import json
import os

R = json.load(open("dda_forward.json", encoding="utf-8"))
os.makedirs("figures", exist_ok=True)
P = R["points"]
xs = [p["mean_dealt_size"] for p in P]
share = [p["EVPI_share"]["median"] for p in P]
share_lo = [p["EVPI_share"]["ci"][0] for p in P]
share_hi = [p["EVPI_share"]["ci"][1] for p in P]
surv = [p["surv"]["strong"] for p in P]
betas = [p["beta"] for p in P]

W, H = 860, 540
ML, MR, MT, MB = 64, 64, 80, 90
PW, PH = W - ML - MR, H - MT - MB
xmin, xmax = min(xs) - 0.15, max(xs) + 0.15


def X(v):
    return ML + PW * (v - xmin) / (xmax - xmin)


def Yshare(v):     # 右轴/共享 0..1.2
    return MT + PH * (1 - v / 1.2)


out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
       f'font-family="Helvetica,Arial,sans-serif" font-size="13">',
       f'<rect width="{W}" height="{H}" fill="white"/>']
out.append(f'<text x="{W/2}" y="26" font-size="17" text-anchor="middle" font-weight="bold">'
           f'B: DDA 设计曲线 — 难度(生存)可由发牌拧动，但运气含量(EVPI 占比)与之解耦</text>')
out.append(f'<text x="{W/2}" y="46" font-size="11" text-anchor="middle" fill="#555">'
           f'N={R["N"]} · T={R["T"]} · D_oracle={R["D_oracle"]} · scoring={R["scoring"]} · '
           f'发牌族 w∝exp(-β·size)，β:{R["betas"][0]}→{R["betas"][-1]} · per-seed 中位数 + 95% CI</text>')

# 网格 + 左轴(share/surv 同 0..1.2 刻度)
for v in (0, 0.25, 0.5, 0.75, 1.0):
    yy = Yshare(v)
    out.append(f'<line x1="{ML}" y1="{yy}" x2="{ML+PW}" y2="{yy}" stroke="#eee"/>')
    out.append(f'<text x="{ML-6}" y="{yy+4}" text-anchor="end" font-size="11">{int(v*100)}%</text>')
out.append(f'<line x1="{ML}" y1="{Yshare(1.0)}" x2="{ML+PW}" y2="{Yshare(1.0)}" '
           f'stroke="#d33" stroke-width="1.1" stroke-dasharray="4,3"/>')

# x 轴刻度 (mean size + beta)
for p in P:
    xx = X(p["mean_dealt_size"])
    out.append(f'<line x1="{xx}" y1="{MT+PH}" x2="{xx}" y2="{MT+PH+5}" stroke="#999"/>')
    out.append(f'<text x="{xx}" y="{MT+PH+18}" text-anchor="middle" font-size="10">'
               f'{p["mean_dealt_size"]:.2f}</text>')
    out.append(f'<text x="{xx}" y="{MT+PH+31}" text-anchor="middle" font-size="9" fill="#999">'
               f'β={p["beta"]:+.2f}</text>')
out.append(f'<text x="{ML+PW/2}" y="{MT+PH+50}" text-anchor="middle" font-size="11">'
           f'← 善意/易（小块）   E[每块格数]   难/对抗（大块）→</text>')

# β=0 (uniform) 竖参考线
for p in P:
    if abs(p["beta"]) < 1e-9:
        xx = X(p["mean_dealt_size"])
        out.append(f'<line x1="{xx}" y1="{MT}" x2="{xx}" y2="{MT+PH}" stroke="#bbb" '
                   f'stroke-dasharray="3,3"/>')
        out.append(f'<text x="{xx+3}" y="{MT+12}" font-size="10" fill="#888">β=0 均匀(原)</text>')


def polyline(ys, col, wdt=2.5):
    pts = " ".join(f"{X(xs[i])},{ys[i]}" for i in range(len(xs)))
    out.append(f'<polyline points="{pts}" fill="none" stroke="{col}" stroke-width="{wdt}"/>')


# EVPI-share CI 带
band = " ".join(f"{X(xs[i])},{Yshare(share_hi[i])}" for i in range(len(xs)))
band += " " + " ".join(f"{X(xs[i])},{Yshare(share_lo[i])}" for i in range(len(xs)-1, -1, -1))
out.append(f'<polygon points="{band}" fill="#c8623f" opacity="0.13"/>')
# 曲线
polyline([Yshare(v) for v in surv], "#3a78b5")       # 生存率(strong)
polyline([Yshare(v) for v in share], "#c8623f")      # EVPI 占比
for i in range(len(xs)):
    out.append(f'<circle cx="{X(xs[i])}" cy="{Yshare(share[i])}" r="3.2" fill="#c8623f"/>')
    out.append(f'<text x="{X(xs[i])}" y="{Yshare(share[i])-7}" text-anchor="middle" '
               f'font-size="9" fill="#a8482b">{share[i]*100:.0f}</text>')
    out.append(f'<circle cx="{X(xs[i])}" cy="{Yshare(surv[i])}" r="3" fill="#3a78b5"/>')

# 图例
out.append(f'<line x1="{ML+12}" y1="{MT+8}" x2="{ML+34}" y2="{MT+8}" stroke="#c8623f" stroke-width="2.5"/>')
out.append(f'<text x="{ML+40}" y="{MT+12}" font-size="11" fill="#a8482b">EVPI 占比(运气含量)</text>')
out.append(f'<line x1="{ML+200}" y1="{MT+8}" x2="{ML+222}" y2="{MT+8}" stroke="#3a78b5" stroke-width="2.5"/>')
out.append(f'<text x="{ML+228}" y="{MT+12}" font-size="11" fill="#3a78b5">strong 生存率@T</text>')

out.append(f'<text x="{ML}" y="{H-28}" font-size="11" fill="#444">'
           f'正向 DDA（诚实结论）：发牌难度旋钮 β 把 strong 生存率从 0.21 拉到 0.99（强、单调）—'
           f'但 EVPI 占比几乎平（~65–75%，CI 互叠）⇒ **难度与运气含量解耦**。</text>')
out.append(f'<text x="{ML}" y="{H-12}" font-size="11" fill="#444">'
           f'⇒ 运气含量不由"块多大/多难"决定，而由计分结构(A&#39;: 乘法连击 65%→86%)与是否含极端坏块'
           f'(no9: 65%→77%)决定。静态重加权=善意下界；固定 T 保 EVPI 良定义。</text>')
out.append('</svg>')
open("figures/dda_forward.svg", "w", encoding="utf-8").write("\n".join(out))
print("-> figures/dda_forward.svg")
