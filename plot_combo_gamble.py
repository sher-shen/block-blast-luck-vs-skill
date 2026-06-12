"""零依赖 SVG：读 combo_gamble.json，画连击豪赌图：
  ① 赌局价值 G(C)：常数-p(乐观) vs level-conditional 衰减-p(实测) → 深连击是 sucker's bet
  ② strong 实测 streak 长度分布。
跟 plot_endless.py 同风格。"""
import json
import os

R = json.load(open("combo_gamble.json", encoding="utf-8"))
emp = R["empirical"]
os.makedirs("figures", exist_ok=True)
gv = {int(k): v for k, v in emp["gamble_value_const_vs_emp"].items()}
p_emp = emp["p_maintain"]
pbl = {int(k): v for k, v in emp["p_by_level"].items()}

W, H = 940, 540
out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
       f'font-family="Helvetica,Arial,sans-serif" font-size="13">',
       f'<rect width="{W}" height="{H}" fill="white"/>']
out.append(f'<text x="{W/2}" y="26" font-size="17" text-anchor="middle" font-weight="bold">'
           f'连击=press-your-luck 赌局：维持概率随层级衰减 → 深连击是亏本押注</text>')
out.append(f'<text x="{W/2}" y="46" font-size="11" text-anchor="middle" fill="#555">'
           f'strong N={emp["N"]} T={emp["T"]} · p_maintain={p_emp:.2f}(随层级 '
           f'{"→".join(f"{pbl[c]:.2f}" for c in sorted(pbl))}) · combo 奖励占总分='
           f'{emp["combo_share"]*100:.0f}% · 连击均长 {emp["streak_mean"]:.2f} 最长 {emp["streak_max"]}</text>')

# ---- 左图：G_const vs G_emp 柱（assumed 档 u=50）----
ML, MT, PW, PH = 64, 84, 400, 380
out.append(f'<text x="{ML}" y="{MT-10}" font-weight="bold">'
           f'① 期望未来 combo 奖励 G(C)：乐观常数-p vs 实测衰减-p</text>')
Cs = sorted(gv)
gmax = max(gv[c]["G_const"] for c in Cs) * 1.15


def yg(g):
    return MT + PH * (1 - g / gmax)


for g in range(0, int(gmax) + 1, 50):
    yy = yg(g)
    out.append(f'<line x1="{ML}" y1="{yy}" x2="{ML+PW}" y2="{yy}" stroke="#eee"/>')
    out.append(f'<text x="{ML-6}" y="{yy+4}" text-anchor="end" font-size="10">{g}</text>')
gw = PW / len(Cs)
for i, C in enumerate(Cs):
    cx = ML + gw * i + gw / 2
    gc = gv[C]["G_const"]; ge = gv[C]["G_emp"]
    bw = 26
    out.append(f'<rect x="{cx-bw-3}" y="{yg(gc)}" width="{bw}" height="{MT+PH-yg(gc)}" fill="#c9b89b"/>')
    out.append(f'<text x="{cx-bw/2-3}" y="{yg(gc)-3}" text-anchor="middle" font-size="10" '
               f'fill="#7a6a44">{gc:.0f}</text>')
    out.append(f'<rect x="{cx+3}" y="{yg(ge)}" width="{bw}" height="{MT+PH-yg(ge)}" fill="#c0392b"/>')
    out.append(f'<text x="{cx+bw/2+3}" y="{yg(ge)-3}" text-anchor="middle" font-size="10" '
               f'fill="#7a1d14" font-weight="bold">{ge:.0f}</text>')
    out.append(f'<text x="{cx}" y="{MT+PH+18}" text-anchor="middle">C={C}</text>')
    out.append(f'<text x="{cx}" y="{MT+PH+33}" text-anchor="middle" font-size="10" fill="#a33">'
               f'{gc/ge:.1f}×高估</text>')
out.append(f'<rect x="{ML}" y="{MT+PH+44}" width="13" height="13" fill="#c9b89b"/>')
out.append(f'<text x="{ML+18}" y="{MT+PH+55}" font-size="11">G_const(常数 p̄=0.44, 乐观)</text>')
out.append(f'<rect x="{ML+230}" y="{MT+PH+44}" width="13" height="13" fill="#c0392b"/>')
out.append(f'<text x="{ML+248}" y="{MT+PH+55}" font-size="11">G_emp(实测衰减 p_c, 真实)</text>')

# ---- 右图：streak 分布 ----
RX, RPW, RPH = 560, 340, 380
out.append(f'<text x="{RX}" y="{MT-10}" font-weight="bold">② strong 实测连击链长度分布</text>')
sd = {int(k): v for k, v in emp["streak_dist"].items()}
TOPK = max(sd) if sd else 1
maxc = max(sd.values()) if sd else 1
bw = RPW / (TOPK + 1) - 8


def yb(c):
    return MT + RPH * (1 - c / maxc)


for i in range(1, TOPK + 1):
    c = sd.get(i, 0)
    bx = RX + (RPW / (TOPK + 1)) * i
    out.append(f'<rect x="{bx}" y="{yb(c)}" width="{bw}" height="{MT+RPH-yb(c)}" fill="#c08552"/>')
    out.append(f'<text x="{bx+bw/2}" y="{yb(c)-3}" text-anchor="middle" font-size="10">{c}</text>')
    out.append(f'<text x="{bx+bw/2}" y="{MT+RPH+16}" text-anchor="middle" font-size="10">{i}</text>')
out.append(f'<text x="{RX+RPW/2}" y="{MT+RPH+34}" text-anchor="middle" font-size="11">'
           f'连击链长度（连续消除次数）</text>')
out.append(f'<text x="{RX}" y="{MT+RPH+55}" font-size="11" fill="#444">'
           f'87% 的连击链 ≤2 长；最长仅 {emp["streak_max"]} → 长连击罕见</text>')

out.append(f'<text x="{ML}" y="{H-12}" font-size="11" fill="#444">'
           f'读图：combo 占分高达 {emp["combo_share"]*100:.0f}%，但价值来自**大量短链**而非英雄式长连击；'
           f'常数-p 计算器把深连击赌局高估 {gv[3]["G_const"]/gv[3]["G_emp"]:.0f}×（维持概率随层级 0.52→0.16 速降）。</text>')
out.append('</svg>')
open("figures/combo_gamble.svg", "w", encoding="utf-8").write("\n".join(out))
print("-> figures/combo_gamble.svg")
