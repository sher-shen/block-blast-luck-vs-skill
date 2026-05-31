"""零依赖 SVG：读 survival.json + channelB.json，产出两张"运气两通道"图。
 fig3: 存活曲线(通道A 存活运气) — strong/blind/seer 活过 t 轮的比例 + seer hazard 上界。
 fig4: EVPI 分解(通道B 计分运气) — 各 horizon 的 raw=EVPI(信息)+procedure(搜索) 堆叠条 + CI。
"""
import json
import os

SV = json.load(open("survival.json"))
CH = json.load(open("channelB.json"))
os.makedirs("figures", exist_ok=True)


def svg_open(w, h):
    return [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" '
            f'font-family="Helvetica,Arial,sans-serif" font-size="13">',
            f'<rect width="{w}" height="{h}" fill="white"/>']


def text(x, y, s, size=13, anchor="start", color="#222", weight="normal"):
    return (f'<text x="{x}" y="{y}" font-size="{size}" text-anchor="{anchor}" '
            f'fill="{color}" font-weight="{weight}">{s}</text>')


# ---------- fig3: 存活曲线 ----------
def fig3():
    W, H = 760, 420
    s = svg_open(W, H)
    x0, x1, y0, y1 = 80, 700, 70, 330
    Ts = SV["Ts"]; cur = SV["curves"]
    tmax = max(Ts)
    def X(t): return x0 + t / tmax * (x1 - x0)
    def Y(p): return y1 - p * (y1 - y0)
    s.append(text(W/2, 28, "通道A 存活运气：活过 t 轮的比例（固定 horizon）", 16, "middle", "#111", "bold"))
    s.append(text(W/2, 48, "有前瞻(seer)几乎不死 → 死亡基本可避免=技能；不可约运气=罕见杀手序列",
                  12, "middle", "#777"))
    for gp in (0, 0.25, 0.5, 0.75, 1.0):
        gy = Y(gp)
        s.append(f'<line x1="{x0}" y1="{gy}" x2="{x1}" y2="{gy}" stroke="#eee"/>')
        s.append(text(x0 - 6, gy + 4, f"{gp:.2f}", 11, "end", "#999"))
    for t in Ts:
        s.append(text(X(t), y1 + 18, str(t), 11, "middle", "#999"))
    s.append(text(W/2, y1 + 40, "存活轮数 t", 12, "middle", "#555"))
    cols = {"seer": ("#e07b39", "seer 先知(看真未来)"),
            "strong": ("#2f6db5", "strong 在线最强"),
            "blind": ("#9bbce0", "blind 前瞻(采样未来)")}
    for k in ("seer", "strong", "blind"):
        pts = " ".join(f"{X(t):.1f},{Y(p):.1f}" for t, p in zip(Ts, cur[k]))
        s.append(f'<polyline points="{pts}" fill="none" stroke="{cols[k][0]}" stroke-width="2.5"/>')
        for t, p in zip(Ts, cur[k]):
            s.append(f'<circle cx="{X(t):.1f}" cy="{Y(p):.1f}" r="3" fill="{cols[k][0]}"/>')
    yy = y0 + 6
    for k in ("seer", "strong", "blind"):
        s.append(f'<rect x="{x0+12}" y="{yy-10}" width="14" height="6" fill="{cols[k][0]}"/>')
        s.append(text(x0 + 32, yy, cols[k][1], 12, "start", "#444"))
        yy += 18
    hz = SV.get("seer_hazard_point_per_round", SV.get("seer_hazard_ub_per_round"))
    ub = SV.get("seer_hazard_poisson_ub95_per_round")
    s.append(text(x1, y0 + 6, f"seer 死亡 hazard ≈ {hz:.0e}/轮", 12, "end", "#e07b39", "bold"))
    if ub:
        s.append(text(x1, y0 + 24, f"(95% 上界 {ub:.0e}/轮)", 11, "end", "#e07b39"))
    s.append('</svg>')
    open("figures/fig3_survival.svg", "w").write("\n".join(s))


# ---------- fig4: EVPI 分解 ----------
def fig4():
    W, H = 760, 420
    s = svg_open(W, H)
    x0, y0, y1 = 110, 80, 330
    Ts = CH["Ts"]
    bars = [(f"T{T}", CH[f"T{T}"]) for T in Ts]
    maxraw = max(b[1]["raw_gap_seer_strong"]["mean"] for b in bars) * 1.25
    def Y(v): return y1 - v / maxraw * (y1 - y0)
    s.append(text(W/2, 28, "通道B 计分运气：先知的得分优势如何切分", 16, "middle", "#111", "bold"))
    s.append(text(W/2, 48, "raw(seer−strong) = EVPI 信息价值(seer−blind) + procedure 搜索(blind−strong)",
                  12, "middle", "#777"))
    for gv in range(0, int(maxraw) + 1, 500):
        gy = Y(gv)
        s.append(f'<line x1="{x0}" y1="{gy}" x2="{W-40}" y2="{gy}" stroke="#eee"/>')
        s.append(text(x0 - 8, gy + 4, str(gv), 11, "end", "#999"))
    bw = 120; gap = (W - 40 - x0 - bw * len(bars)) / (len(bars) + 1)
    x = x0 + gap
    for label, d in bars:
        evpi = d["EVPI_vs_marginalizer"]["mean"]
        proc = d["procedure_cost"]["mean"]
        raw = d["raw_gap_seer_strong"]["mean"]
        share = d["evpi_share_per_seed_median"] * 100
        # 堆叠：下=EVPI(橙) 上=procedure(蓝)
        s.append(f'<rect x="{x}" y="{Y(evpi):.1f}" width="{bw}" height="{y1-Y(evpi):.1f}" fill="#e07b39"/>')
        s.append(f'<rect x="{x}" y="{Y(evpi+proc):.1f}" width="{bw}" height="{Y(evpi)-Y(evpi+proc):.1f}" fill="#2f6db5"/>')
        # EVPI CI 须
        ci = d["EVPI_vs_marginalizer"]["mean_ci"]
        cx = x + bw / 2
        s.append(f'<line x1="{cx}" y1="{Y(ci[0]):.1f}" x2="{cx}" y2="{Y(ci[1]):.1f}" stroke="#7a3" stroke-width="2"/>')
        s.append(text(x + bw/2, Y(evpi)+18, f"EVPI {evpi:.0f}", 11, "middle", "#fff", "bold"))
        s.append(text(x + bw/2, Y(evpi+proc)-6, f"+search {proc:.0f}", 11, "middle", "#2f6db5"))
        s.append(text(x + bw/2, y1 + 18, f"{label} (n={d['cohort_n']})", 12, "middle", "#444", "bold"))
        s.append(text(x + bw/2, y1 + 36, f"信息占比 {share:.0f}%", 12, "middle", "#e07b39", "bold"))
        x += bw + gap
    s.append(text(W/2, y1 + 64,
                  "信息价值(运气)占先知得分优势的 ~57–69%；其余=前瞻搜索本身", 12, "middle", "#555"))
    s.append('</svg>')
    open("figures/fig4_evpi.svg", "w").write("\n".join(s))


# ---------- fig5: 4×4 精确 DP 锚点 ----------
def fig5():
    import os.path
    if not os.path.exists("dp4.json"):
        return
    D = json.load(open("dp4.json"))
    W, H = 760, 380
    s = svg_open(W, H)
    x0, x1, y0, y1 = 90, 700, 80, 300
    m = D["means"]
    bars = [("greedy 近视", m["greedy"], "#9bbce0"),
            ("online 最优 V*", D["V_star_empty"], "#2f6db5"),
            ("offline 上帝视角", m["offline_opt"], "#e07b39")]
    hi = max(b[1] for b in bars) * 1.18
    def X(v): return x0 + v / hi * (x1 - x0)
    s.append(text(W/2, 28, "4×4 精确 DP 锚点（可解类比；γ=0.95，1块/回合，线性计分）",
                  15, "middle", "#111", "bold"))
    s.append(text(W/2, 48, "唯一能算真·最优的棋盘：近视启发式离最优多远 + 精确信息价值",
                  12, "middle", "#777"))
    y = 70; bh = 54
    for (nm, val, col) in bars:
        s.append(f'<rect x="{x0}" y="{y}" width="{X(val)-x0:.1f}" height="{bh-16}" fill="{col}" rx="3"/>')
        s.append(text(x0 - 8, y + (bh-16)/2 + 2, nm, 13, "end", "#333"))
        s.append(text(X(val) + 6, y + (bh-16)/2 + 2, f"{val:.2f}", 12, "start", "#333", "bold"))
        y += bh
    voi = D["discounted_VOI"]; hg = D["heuristic_gap"]
    ratio = m["online_opt"] / m["offline_opt"] * 100
    s.append(text(x0, y + 6,
                  f"① 近视贪心 ≈ {D['greedy_optimality_ratio']*100:.0f}% 最优（缺口 {hg['mean']:.1f} "
                  f"[{hg['ci'][0]:.1f},{hg['ci'][1]:.1f}]）→ 启发式在线已近最优",
                  12, "start", "#2f6db5", "bold"))
    s.append(text(x0, y + 28,
                  f"② online 最优只兑现上帝视角的 {ratio:.0f}%；discounted VOI = "
                  f"{voi['mean']:.1f} [{voi['ci'][0]:.1f},{voi['ci'][1]:.1f}] → 信息价值主导",
                  12, "start", "#e07b39", "bold"))
    s.append(text(x0, y + 50, "（类比，非 8×8 标定；不建模 combo 运气 / 3块重排）",
                  11, "start", "#999"))
    s.append('</svg>')
    open("figures/fig5_dp4.svg", "w").write("\n".join(s))


fig3(); fig4(); fig5()
print("写出 figures/fig3_survival.svg, fig4_evpi.svg, fig5_dp4.svg")
