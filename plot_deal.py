"""零依赖 SVG：读 deal_audit.json，画各发牌变体的生存 + 总分对比条形图。"""
import json
import os

R = json.load(open("deal_audit.json", encoding="utf-8"))
os.makedirs("figures", exist_ok=True)
ORDER = ["uniform", "no9", "no_hard", "no_ge5", "small_bias"]
LABEL = {"uniform": "均匀38(现)", "no9": "去3x3", "no_hard": "去≥6格",
         "no_ge5": "去≥5格", "small_bias": "强偏小块"}
order = [v for v in ORDER if v in R]

W, H = 820, 460
ML, MR, MT, MB = 64, 64, 56, 70
PW, PH = W - ML - MR, H - MT - MB
n = len(order)
gw = PW / n

smax = max(R[v]["surv"] for v in order) * 1.18
tmax = max(R[v]["total"] for v in order) * 1.18


def sy(v):
    return MT + PH * (1 - v / smax)


def ty(v):
    return MT + PH * (1 - v / tmax)


out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
       f'font-family="Helvetica,Arial,sans-serif" font-size="13">',
       f'<rect width="{W}" height="{H}" fill="white"/>']
out.append(f'<text x="{W/2}" y="26" font-size="17" text-anchor="middle" font-weight="bold">'
           f'Step B: 发牌分布敏感性 — 善意发牌 vs 均匀38 (M={len(R["uniform"]["survs"])}, T=500, 最强策略)</text>')
out.append(f'<text x="{W/2}" y="44" font-size="12" text-anchor="middle" fill="#666">'
           f'蓝=生存回合(左轴) 橙=总分(右轴) · 误差棒=16-block SE · ✂=撞T=500封顶比例(数值被截断,为下界)</text>')

# 轴
out.append(f'<line x1="{ML}" y1="{MT+PH}" x2="{ML+PW}" y2="{MT+PH}" stroke="#333"/>')
out.append(f'<line x1="{ML}" y1="{MT}" x2="{ML}" y2="{MT+PH}" stroke="#36c"/>')
out.append(f'<line x1="{ML+PW}" y1="{MT}" x2="{ML+PW}" y2="{MT+PH}" stroke="#e80"/>')
out.append(f'<text x="18" y="{MT+PH/2}" text-anchor="middle" fill="#36c" '
           f'transform="rotate(-90 18 {MT+PH/2})">生存回合 surv</text>')
out.append(f'<text x="{W-16}" y="{MT+PH/2}" text-anchor="middle" fill="#e80" '
           f'transform="rotate(-90 {W-16} {MT+PH/2})">总分</text>')

for i, v in enumerate(order):
    cx = ML + gw * (i + 0.5)
    r = R[v]
    # 生存柱(蓝, 左半)
    bw = gw * 0.30
    x1 = cx - bw - 3
    out.append(f'<rect x="{x1}" y="{sy(r["surv"])}" width="{bw}" height="{MT+PH-sy(r["surv"])}" fill="#69c"/>')
    se = r.get("surv_se", 0)
    out.append(f'<line x1="{x1+bw/2}" y1="{sy(r["surv"]-se)}" x2="{x1+bw/2}" y2="{sy(r["surv"]+se)}" stroke="#234" stroke-width="1.4"/>')
    out.append(f'<text x="{x1+bw/2}" y="{sy(r["surv"])-6}" text-anchor="middle" font-size="11" fill="#36c">{r["surv"]:.0f}</text>')
    # 总分柱(橙, 右半)
    x2 = cx + 3
    out.append(f'<rect x="{x2}" y="{ty(r["total"])}" width="{bw}" height="{MT+PH-ty(r["total"])}" fill="#f93"/>')
    tse = r.get("total_se", 0)
    out.append(f'<line x1="{x2+bw/2}" y1="{ty(r["total"]-tse)}" x2="{x2+bw/2}" y2="{ty(r["total"]+tse)}" stroke="#a40" stroke-width="1.4"/>')
    out.append(f'<text x="{x2+bw/2}" y="{ty(r["total"])-6}" text-anchor="middle" font-size="11" fill="#e80">{r["total"]:.0f}</text>')
    # x 标签
    out.append(f'<text x="{cx}" y="{MT+PH+20}" text-anchor="middle" font-weight="bold">{LABEL.get(v,v)}</text>')
    M = len(r["survs"])
    cen = r["censored"]
    if cen:
        out.append(f'<text x="{cx}" y="{MT+PH+38}" text-anchor="middle" font-size="11" fill="#c33">✂ {cen}/{M} 封顶</text>')
    # 相对倍数
    mult = r["surv"] / R["uniform"]["surv"]
    if v != "uniform":
        out.append(f'<text x="{cx}" y="{MT+PH+54}" text-anchor="middle" font-size="11" fill="#197">surv {mult:.1f}×</text>')

out.append('</svg>')
open("figures/deal_audit.svg", "w", encoding="utf-8").write("\n".join(out))
print("-> figures/deal_audit.svg")
