"""零依赖 SVG：读 calibrate_scoring.json，画计分校准三档对比 + "计分非真凶"结论。"""
import json
import os

R = json.load(open("calibrate_scoring.json", encoding="utf-8"))
os.makedirs("figures", exist_ok=True)
rows = R  # list of dicts in cfg order
LAB = [f'{r["profile"]}\n{r["deal"]}' for r in rows]

W, H = 720, 440
ML, MR, MT, MB = 70, 20, 60, 78
PW, PH = W - ML - MR, H - MT - MB
n = len(rows)
gw = PW / n
tmax = max(r["total"] for r in rows) * 1.2


def ty(v):
    return MT + PH * (1 - v / tmax)


out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
       f'font-family="Helvetica,Arial,sans-serif" font-size="13">',
       f'<rect width="{W}" height="{H}" fill="white"/>']
out.append(f'<text x="{W/2}" y="26" font-size="17" text-anchor="middle" font-weight="bold">'
           f'Step C: 计分校准 — "真实近似档" vs "假设档" (同策略, T=500)</text>')
out.append(f'<text x="{W/2}" y="44" font-size="12" text-anchor="middle" fill="#666">'
           f'real/assumed(同uniform发牌)=0.71× ⇒ 同量级, 计分公式不是"分低"真凶</text>')

out.append(f'<line x1="{ML}" y1="{MT+PH}" x2="{ML+PW}" y2="{MT+PH}" stroke="#333"/>')
out.append(f'<line x1="{ML}" y1="{MT}" x2="{ML}" y2="{MT+PH}" stroke="#333"/>')
out.append(f'<text x="20" y="{MT+PH/2}" text-anchor="middle" '
           f'transform="rotate(-90 20 {MT+PH/2})">总分 total</text>')

cols = ["#789", "#5a8", "#3a6"]
for i, r in enumerate(rows):
    cx = ML + gw * (i + 0.5)
    bw = gw * 0.5
    x = cx - bw / 2
    out.append(f'<rect x="{x}" y="{ty(r["total"])}" width="{bw}" height="{MT+PH-ty(r["total"])}" '
               f'fill="{cols[i%len(cols)]}"/>')
    se = r["total_se"]
    out.append(f'<line x1="{cx}" y1="{ty(r["total"]-se)}" x2="{cx}" y2="{ty(r["total"]+se)}" stroke="#222"/>')
    out.append(f'<text x="{cx}" y="{ty(r["total"])-8}" text-anchor="middle" font-size="12" '
               f'font-weight="bold">{r["total"]:.0f}</text>')
    out.append(f'<text x="{cx}" y="{MT+PH+18}" text-anchor="middle" font-size="12" font-weight="bold">'
               f'{r["profile"]}</text>')
    out.append(f'<text x="{cx}" y="{MT+PH+34}" text-anchor="middle" font-size="11" fill="#555">'
               f'发牌={r["deal"]}</text>')
    out.append(f'<text x="{cx}" y="{MT+PH+50}" text-anchor="middle" font-size="11" fill="#36c">'
               f'surv={r["surv"]:.0f}</text>')
    M = len(r["survs"])
    if r["censored"]:
        out.append(f'<text x="{cx}" y="{MT+PH+66}" text-anchor="middle" font-size="10" fill="#c33">'
                   f'✂{r["censored"]}/{M}封顶(下界)</text>')

out.append('</svg>')
open("figures/calibrate_scoring.svg", "w", encoding="utf-8").write("\n".join(out))
print("-> figures/calibrate_scoring.svg")
