"""零依赖 SVG：读 evpi_vla.json，画"换在线天花板(strong->vla)后运气占比抬升"+ 分数阶梯。
跟 plot_endless.py 同风格(纯标准库 SVG -> figures/)。"""
import json
import os

R = json.load(open("evpi_vla.json", encoding="utf-8"))
os.makedirs("figures", exist_ok=True)
TS = R["Ts"]

W, H = 920, 540
out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
       f'font-family="Helvetica,Arial,sans-serif" font-size="13">',
       f'<rect width="{W}" height="{H}" fill="white"/>']
out.append(f'<text x="{W/2}" y="26" font-size="17" text-anchor="middle" font-weight="bold">'
           f'verdict(i): 换在线天花板 strong→vla(2924) → 运气(信息价值)占比抬升</text>')
cfg = R["vla_cfg"]
out.append(f'<text x="{W/2}" y="46" font-size="11" text-anchor="middle" fill="#555">'
           f'N={R["N"]} · D_oracle={R["D_oracle"]} · vla={cfg["D"]}/{cfg["S"]}/{cfg["B"]} '
           f'· 冻结 intersection-of-survivors cohort · per-seed 配对中位数 + bootstrap 95% CI</text>')

# ---- 左图：占比柱(strong-denom vs vla-denom，按 T) ----
ML, MT = 60, 80
PW, PH = 380, 380
x0 = ML
out.append(f'<text x="{x0}" y="{MT-8}" font-weight="bold">① 运气占比 EVPI/raw（cohort4 同种子）</text>')


def y_share(v):  # 0..1.45 -> px
    return MT + PH * (1 - v / 1.45)


# 网格
for v in (0, 0.25, 0.5, 0.75, 1.0, 1.25):
    yy = y_share(v)
    out.append(f'<line x1="{x0}" y1="{yy}" x2="{x0+PW}" y2="{yy}" stroke="#eee"/>')
    out.append(f'<text x="{x0-6}" y="{yy+4}" text-anchor="end" font-size="11">{int(v*100)}%</text>')
# 100% 参考线
out.append(f'<line x1="{x0}" y1="{y_share(1.0)}" x2="{x0+PW}" y2="{y_share(1.0)}" '
           f'stroke="#d33" stroke-width="1.5" stroke-dasharray="4,3"/>')
out.append(f'<text x="{x0+PW}" y="{y_share(1.0)-4}" text-anchor="end" font-size="10" fill="#d33">'
           f'100%=残差全是信息价值</text>')

gw = PW / len(TS)
for i, T in enumerate(TS):
    d = R[f"T{T}"]
    cx = x0 + gw * i + gw / 2
    s_strong = d["share_strong_denom_cohort4"]["median"]
    s_vla = d["share_vla_denom_cohort4_PRIMARY"]["median"]
    ci_v = d["share_vla_denom_cohort4_PRIMARY"]["ci"]
    bw = 38
    # strong-denom 柱
    out.append(f'<rect x="{cx-bw-4}" y="{y_share(s_strong)}" width="{bw}" '
               f'height="{MT+PH-y_share(s_strong)}" fill="#9bb7d4"/>')
    out.append(f'<text x="{cx-bw/2-4}" y="{y_share(s_strong)-4}" text-anchor="middle" '
               f'font-size="11" fill="#36506b">{s_strong*100:.0f}%</text>')
    # vla-denom 柱
    out.append(f'<rect x="{cx+4}" y="{y_share(s_vla)}" width="{bw}" '
               f'height="{MT+PH-y_share(s_vla)}" fill="#e1843a"/>')
    out.append(f'<text x="{cx+bw/2+4}" y="{y_share(s_vla)-4}" text-anchor="middle" '
               f'font-size="11" fill="#a85515" font-weight="bold">{s_vla*100:.0f}%</text>')
    # vla CI 须
    out.append(f'<line x1="{cx+bw/2+4}" y1="{y_share(ci_v[0])}" x2="{cx+bw/2+4}" '
               f'y2="{y_share(ci_v[1])}" stroke="#a85515" stroke-width="1.5"/>')
    out.append(f'<text x="{cx}" y="{MT+PH+18}" text-anchor="middle">T={T}</text>')
    out.append(f'<text x="{cx}" y="{MT+PH+34}" text-anchor="middle" font-size="10" fill="#888">'
               f'n={d["cohort4_n"]}</text>')
# 图例
out.append(f'<rect x="{x0}" y="{MT+PH+44}" width="13" height="13" fill="#9bb7d4"/>')
out.append(f'<text x="{x0+18}" y="{MT+PH+55}" font-size="11">strong 当分母(旧)</text>')
out.append(f'<rect x="{x0+150}" y="{MT+PH+44}" width="13" height="13" fill="#e1843a"/>')
out.append(f'<text x="{x0+168}" y="{MT+PH+55}" font-size="11">vla 当分母(新, verdict i)</text>')

# ---- 右图：分数阶梯(cohort4 中位数, T=50) ----
RX = 540
RPW, RPH = 320, 380
out.append(f'<text x="{RX}" y="{MT-8}" font-weight="bold">② 分数阶梯 cohort4 中位数 (T=50)</text>')
d50 = R["T50"]["score_median_cohort4"]
players = [("strong", "#9bb7d4", "strong 2364档"), ("blind", "#b9a0d4", "blind 边缘化"),
           ("vla", "#e1843a", "vla 最强可玩"), ("seer", "#5aa469", "seer 完美前瞻")]
vmax = max(d50[p] for p, _, _ in players) * 1.12


def y_sc(v):
    return MT + RPH * (1 - v / vmax)


for v in range(0, int(vmax) + 1, 1000):
    yy = y_sc(v)
    out.append(f'<line x1="{RX}" y1="{yy}" x2="{RX+RPW}" y2="{yy}" stroke="#eee"/>')
    out.append(f'<text x="{RX-6}" y="{yy+4}" text-anchor="end" font-size="11">{v}</text>')
bw2 = RPW / len(players) - 16
for i, (p, col, lab) in enumerate(players):
    v = d50[p]
    bx = RX + (RPW / len(players)) * i + 8
    out.append(f'<rect x="{bx}" y="{y_sc(v)}" width="{bw2}" height="{MT+RPH-y_sc(v)}" fill="{col}"/>')
    out.append(f'<text x="{bx+bw2/2}" y="{y_sc(v)-4}" text-anchor="middle" font-size="11" '
               f'font-weight="bold">{v:.0f}</text>')
    out.append(f'<text x="{bx+bw2/2}" y="{MT+RPH+16}" text-anchor="middle" font-size="10">{p}</text>')
# 标注 EVPI(seer-blind) 与 seer-vla 残差
ev = R["T50"]["EVPI_seer_blind_cohort4"]["median"]
rv = R["T50"]["raw_gap_seer_vla_cohort4"]["median"]
out.append(f'<text x="{RX}" y="{MT+RPH+44}" font-size="11" fill="#333">'
           f'seer−blind(EVPI)={ev:.0f} · seer−vla(残差=对最强可玩玩家的不可约运气)={rv:.0f}</text>')
out.append(f'<text x="{RX}" y="{MT+RPH+60}" font-size="11" fill="#a85515">'
           f'vla&gt;blind ⇒ procedure 技能通道被 vla 吃满，残差≈纯信息价值</text>')

# 底注
out.append(f'<text x="{ML}" y="{H-10}" font-size="11" fill="#444">'
           f'诚实口径：vla 是无真未来的强边缘化器，已超过 blind ⇒ vla-分母占比可 &gt;100%（含义见图②）。'
           f'占比对在线强度单调上偏保守，故"只升"= 预注册 verdict(i) 命中。</text>')
out.append('</svg>')
open("figures/evpi_vla.svg", "w", encoding="utf-8").write("\n".join(out))
print("-> figures/evpi_vla.svg")
