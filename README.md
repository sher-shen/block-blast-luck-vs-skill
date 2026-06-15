# Luck as the Value of Hindsight
### Quantitatively decomposing luck and skill in Block Blast-style games via the "value of information"

> **In one sentence**: in 8×8 Block Blast-style games (single-player + random piece draws + survival type), we operationalize **luck** as a computable quantity — **the extra points a player with a "god's-eye view" (knowing the future pieces in advance) scores over an online optimal player who can only see the current state**. This is precisely the **value of the information the player is missing (value of hindsight)**. We then cross-validate with three independent lines of evidence — variance decomposition, a skill ladder, and search convergence — to give a quantitative answer to "how much of this game is really luck."

---

## Why do this / how it differs from existing work

Almost everything online about Block Blast falls into two categories: **(1) "solver" websites that help you clear levels**; **(2) training an RL agent to play the game well** (e.g. the DQN/PPO of [RisticDjordje/BlockBlast-Game-AI-Agent](https://github.com/RisticDjordje/BlockBlast-Game-AI-Agent)). Both ask "**how to play better**."

Meanwhile, the academic skill-vs-luck literature ([Skill vs Chance, arXiv:2410.14363](https://arxiv.org/pdf/2410.14363), [Geometry of Games, arXiv:2511.11611](https://arxiv.org/pdf/2511.11611)) is almost entirely about **multiplayer** games and rating systems.

The three closest references each still differ mechanistically (and therefore do not constitute prior art):
- **[Skill or Luck? Return Decomposition via Advantage Functions, arXiv:2402.12874](https://arxiv.org/abs/2402.12874)**:
  uses advantage functions to split a single agent's return into skill/luck. But it decomposes the return of a single trajectory at the **policy–value advantage mechanism level**, whereas we operationalize luck on a dual channel — the **information level** (EVPI = seer−blind) + the **survival level** — and anchor the ceiling with exact DP. The mechanism differs.
- **[Gehnen & Venier, "Tetris Is Not Competitive", FUN 2024](https://drops.dagstuhl.de/entities/document/10.4230/LIPIcs.FUN.2024.16)**:
  the closer academic pillar for channel A (survival luck = irreducible killer sequences); it characterizes "no winning / no infinite-survival strategy in single-player block-puzzles" more directly than [Burgiel](https://www.semanticscholar.org/paper/How-to-lose-at-Tetris-Burgiel/11c12871bfa138fa8bb93a4e5dbcca36c5d214fa).
- **[Difficulty estimation via RL agents, arXiv:2306.14626](https://arxiv.org/abs/2306.14626)**:
  uses RL agent performance as a **proxy for human level difficulty / completion rate** — a precedent for "using agent performance as a difficulty/completion-rate proxy" (related to this repo's direction ②b DDA), **not** a precedent for ②a strong-policy ceiling; do not cite it as the latter.

**The gap**: nobody has framed Block Blast as a "**quantitative luck–skill decomposition**" problem. To our knowledge, this project is the **first to apply an EVPI/survival dual-channel decomposition to single-player random survival-type block-puzzles, anchoring the information-value ceiling with exact DP**. The theoretical anchor comes from an old saying — [*"luck is nothing more than a lack of information"* (Aleph Insights)](https://alephinsights.com/blog/2016/05/skill-and-luck/) — which we **operationalize and compute**.

---

## Game and modeling

- 8×8 board, no gravity; each round deals **3 pieces**, all of which must be placed before a refresh; filling a complete row/column clears it.
- **A 38-piece catalog** (including rotational orientations, uniformly equiprobable, see [`pieces.py`](pieces.py)).
- **Non-linear scoring** ([`scoring.py`](scoring.py)): clearing multiple lines at once grows by triangular numbers, with rewards for combos and all-clears — these rewards specifically reward "cross-hand layout," which is the key to the analysis.
- Two engine versions: a readable reference implementation [`sim.py`](sim.py) (list-based board) + a high-speed [`fast.py`](fast.py) (64-bit bitboard, bitwise can_place / line clearing / heuristics).

Four players (skill from low to high): `random` (random placement) → `greedy` (per-piece greedy) → `strong` (in-hand beam search, combo-aware throughout) → `lookahead` (beam candidates + flat Monte-Carlo rollout, λ=1, CRN-paired).

---

## Method (three independent lines of evidence + one new metric)

1. **Variance decomposition (ANOVA)**: feed the same batch of random seeds to different players, splitting total-score variance into
   *skill (player main effect)* / *luck (seed main effect)* / *interaction*.
2. **Skill ladder**: use ε-greedy to continuously interpolate between random↔strong, creating a "skill axis,"
   defining the **crossover point** = the score where "gap to the ceiling = single-game luck fluctuation σ" —
   **gap > σ → skill-dominated; gap < σ → luck-dominated**. This answers "below what score it's about practice, above it about fate."
3. **Search convergence**: keep strengthening the player and watch whether the ceiling stops rising. If lookahead ≈ strong, the skill limit has nearly been reached, and the remaining variance is irreducible luck.
4. **[Novel metric] value of information = luck** (see [`oracle_analysis.py`](oracle_analysis.py)):
   let a **seer** use the **true future pieces** during rollout, and compare it against a **structurally identical blind player that can only see the future _distribution_ (averaging over sampled futures)** — the only difference between the two is "whether they know this game's true future." Their score difference
   = **expected value of perfect information (EVPI)** = the quantification of luck.

> ⚠️ **Important correction (2026-05-31, after three rounds of independent review)**: an early version used
> `1 − strong_mean/oracle_mean ≈ 78%` as the luck share. We later found that once the oracle was upgraded to beam-rollout + true future, it became
> **nearly unkillable** — in unbounded games it could survive tens of thousands of rounds, with the score rolling up to 1e5–1e6. As a result the "score difference" was **dominated by survival length** and **inherently unbounded**, so the denominator of that 78% was ill-defined and has been **retracted** (see [`results.json`](results.json) `oracle_RETRACTED`).
> The correct approach splits luck into **two channels**, both measured under a **fixed horizon T** (all players run only T rounds so scores are comparable):
>
> - **Channel A · survival luck**: the fraction of each player surviving past round t. The seer almost never dies → "death" is basically avoidable = skill;
>   irreducible survival luck = the rare [Burgiel killer sequences](https://www.semanticscholar.org/paper/How-to-lose-at-Tetris-Burgiel/11c12871bfa138fa8bb93a4e5dbcca36c5d214fa),
>   upper-bounded by the seer's **per-round death hazard** (the true optimum survives ≥ the seer).
> - **Channel B · scoring luck (EVPI)** = `seer − blind`, **computed only on the cohort where strong/blind/seer all survive to T**
>   (otherwise a dead player's score is frozen, leaking survival differences back into scoring). Then split the seer's total advantage with sign:
>   `raw(seer−strong) = EVPI(information) + procedure(blind−strong, the search itself)`, **without clipping**.
>
> Key traps caught and fixed across the three review rounds: ① blind may be < strong (*rollout regression*) → procedure must be signed, with the floor at strong; ② death-frozen scores pollute scoring → condition on the cohort; ③ the cohort keeps only the "easy seeds where strong can survive" → the computed luck share is a **lower bound** (the luck of the hardest top-end games is counted in channel A); ④ using a "confidently wrong" anti player would overestimate the value of information → the headline uses `seer−blind`; ⑤ EVPI is biased at both ends → report only "≈," backed by **EVPI flattening with sample count S and lookahead depth D**.

---

## Conclusions

![players](figures/fig1_players.svg)
![ladder](figures/fig2_ladder.svg)

Data: 38 pieces, variance/ladder over 200 seeds ([`results.json`](results.json));
luck two channels over N=120 seeds, D=3, fixed horizon ([`survival.json`](survival.json) / [`channelB.json`](channelB.json)).

**① The skill floor is enormous — this is the "game of skill" face.**
Random placement averages **57**, skilled play (strong) **6453**, a gap of **≈113×**. Just "not playing stupidly" decides two orders of magnitude.

**② Among skilled players, luck ≈ skill — this is the "game of luck" face.**
Variance decomposition (greedy vs strong): **skill 34% / luck 34% / interaction 32%**.
And that 32% interaction term is "**the good-draw opportunities only a skilled player can cash in**," which is essentially luck-leaning.

**③ The dividing line is not a single cut, but sits very high.**
The skill ladder shows: scores **below ~3400 (the crossover point) are skill-dominated essentially throughout** (gap/σ falls all the way from 139 down to ~6, all >1),
and only near the ceiling does it flip to luck-dominated. **So "low score = not enough practice, not bad luck" holds for the vast majority of skill levels.**

**④ Luck splits into two channels (fixed horizon, N=120, D=3, after three rounds of review).**

![survival](figures/fig3_survival.svg)
![evpi](figures/fig4_evpi.svg)

- **Channel A · survival luck ≈ minimal.** The fraction of the seer surviving past t is **≈ 1.00 throughout** (0.99 at t=120),
  with a **per-round death hazard point estimate of 7×10⁻⁵/round** (only 1 death in 14365 at-risk rounds; **one-sided 95% Poisson upper bound 3.3×10⁻⁴/round ≈ ≤1 death per 3028 rounds** —
  the sampling uncertainty of a single event is large, so we report a point estimate + upper bound rather than treating the point estimate as a hard upper bound). There is also an independent **modeling** upper bound: "the true optimum survives ≥ seer," so the true killer-sequence rate is even smaller. By contrast, online strong drops from
  0.91 (t=20) all the way to 0.38 (t=120), and blind is worse (0.20). **Implication: with lookahead the game is almost impossible to lose → an ordinary player's "death" is overwhelmingly an avoidable technical issue, not luck; the only irreducible survival luck is the rare Burgiel must-die draw.**
- **Channel B · scoring luck (EVPI) ≈ 60% of the seer's score advantage.** On the cohort where "all survive to T" (the share being the per-seed
  paired median of `(seer−blind)/(seer−strong)`, **defined only on seeds where seer>strong**, with a bootstrap 95% CI):
  - T=40 (n=54): `raw 1442 = EVPI 943 [816,1082] + procedure 499`, information share **69% [53,73]**.
  - T=50 (n=41): `raw 1856 = EVPI 1180 [1020,1356] + procedure 675`, information share **65% [54,71]**.
  - T=60 (n=34): `raw 2281 = EVPI 1427 [1221,1662] + procedure 854`, information share **57% [54,74]**.
  - That is, of the seer's score advantage over the strongest online player, **~57–69% is the pure value of "knowing the future" (luck)**, the rest being the credit of the lookahead search itself. This ratio is **stable** across sample count S∈{4..32} and lookahead depth D (saturating at D≥3).
  - ⚠️ The **denominator** of the share, `seer−strong`, leans on the assumption that "strong = the true online ceiling"; if there were a stronger online player, the denominator would shrink and the information share would **rise** (so the current value is **conservatively biased downward** with respect to strong's strength). This is exactly the assumption that direction ① (a learned value function) sets out to stress-test.

**⑤ The 4×4 exact-DP anchor (a solvable analogue, after three rounds of review).**

![dp4](figures/fig5_dp4.svg)

8×8 cannot be solved exactly, but **4×4 can** — yielding two things unobtainable on 8×8 (`dp4.py`, M=200 sequences, γ=0.95):

- **The myopic heuristic is already near online optimal**: greedy reaches **≈86%** of the true optimum (value iteration `V*`) (gap 0.82 [0.37,1.27] discounted points)
  → corroborating that "using beam-strong as the 8×8 online baseline" is reasonable.
- **The value of information dominates**: even a **provably optimal** online policy only realizes **≈25%** of the "god's-eye view (offline DP)";
  **discounted VOI = 12.6 [12.0,13.3]** (≈3× the online optimum) → on a small, exactly solvable, combo-free board,
  **"knowing the future" is still an overwhelming value** — independently corroborating the 8×8 "information = luck ceiling" main line.

> This is an **analogue, not a calibration**: 1 piece/round (not a 3-piece hand), linear scoring (no combo/all-clear), γ=0.95 discount.
> So it does **not** reproduce the 8×8 57–69%, and does **not** model combo luck or 3-piece rearrangement skill — it only independently confirms two things:
> the heuristic is near online optimal, and the value of information is huge. Correctness is guaranteed by the hard assertion `mean(online)=V*(empty)` (M=200 passing).

**Summary**: Block Blast is a game where **"skill sets the floor, luck sets the ceiling,"** but the ceiling's "luck" has two faces —
**survival is almost entirely skill** (near-unkillable under perfect lookahead, the only real luck being rare killer sequences), while **given survival, scoring more is about 60% down to draw luck** (EVPI is 57–69% of the seer's advantage). At low-to-mid levels you compete on strategy (a **113×** floor), at the top you compete on draw luck in the "scoring" dimension. It is a different paradigm from Gomoku (luck = 0, with a winning strategy): here there is **no winning strategy**, and even infinite survival is denied by
[Burgiel-style "How to lose at Tetris"](https://www.semanticscholar.org/paper/How-to-lose-at-Tetris-Burgiel/11c12871bfa138fa8bb93a4e5dbcca36c5d214fa)
killer sequences — it's just that such sequences are **extremely rare**.

---

## Play it yourself

- A self-contained, zero-dependency playable version lives in `play.html` — just open it in any browser.
- Click a piece to pick it up, move the pointer, then click the board to drop it (no dragging/holding needed); press Esc to cancel.
- It mirrors the exact mechanics of `sim.py` / `scoring.py` / `pieces.py` (the same 38-piece catalog, the non-linear scoring, instant row/column clears, the combo counter, and the game-over condition), so it is a faithful way to sanity-check the rules by hand.
- A checkbox lets you exclude the plus-pentomino (`pent_plus`), and a dropdown switches between the `assumed` and `real_approx` scoring tables.

---

## Honest limitations

- **EVPI is "≈" not "="**: the seer looking ahead only D hands (saturating at D≥3) is a **lower bound** on the true offline optimum; blind uses S samples to approximate "distribution-optimal" play and still carries Monte-Carlo noise. The two biases point in opposite directions, so EVPI reports only an order of magnitude + CI, backed by "flattening with S/D."
- **cohort selection bias → the scoring-luck share is a lower bound**: channel B is computed only on seeds where "strong can also survive to T," and these are the **easier games**; the luck of top-end hard games (where strong dies early) is counted in **channel A**. So "information share 57–69%" is a **conservative lower bound** on scoring luck.
- **survival hazard has two layers of bounds, which must be kept distinct**: (1) **sampling** uncertainty — the point estimate 7×10⁻⁵/round is based on 1 death, with a one-sided 95% Poisson upper bound of 3.3×10⁻⁴/round;
  (2) **modeling** bound — the true optimum survives ≥ seer, so the true killer-sequence rate is smaller than any of the numbers above. The two point the same way (both saying "luck-induced death is extremely rare") but come from different sources.
- **conclusions depend on the scoring model**: the line_base/combo/all-clear values and the **uniformly equiprobable** dealing are modeling assumptions; the seer scores precisely by compounding combos, and a real game using **adaptive RNG** (gaming you according to the board) would change the ratio. Scoring sensitivity analysis is future work.
- **the 4×4 DP is an analogue, not a calibration** (see §⑤): 1 piece/round + linear scoring + γ discount, three deviations from the real 8×8; it independently confirms "the heuristic is near online optimal" and "the value of information is huge," but does not reproduce the specific 8×8 percentages, nor model combo luck.
- truly approaching the "online ceiling" requires a **learned value function (RL/DQN)**; this project stops at "search has converged + the value of information has been quantified," without training RL.

---

## Reproduce

Zero dependencies, pure Python standard library (including hand-written SVG plotting).

```bash
python3 pieces.py            # view the 38-piece catalog
python3 sim.py 300           # variance decomposition (readable engine)
python3 ladder.py 150        # skill ladder + crossover point
python3 compare.py 20        # paired convergence (greedy-base vs strong-base rollout)
python3 experiments.py 200 24  # variance/ladder/convergence -> results.json
python3 plots.py             # results.json -> figures/fig1,fig2

# luck two channels (novel main line, after three rounds of review; D=3 fixed as plateau by D-sweep)
python3 oracle_analysis.py sweep 24 80      # D-sweep: fix lookahead depth D (survival/score plateau)
python3 oracle_analysis.py sstab 40 3 40    # whether EVPI flattens with sample count S
python3 oracle_analysis.py survival 120 3   # channel A survival curves -> survival.json
python3 oracle_analysis.py channel 120 3 40,50,60  # channel B EVPI decomposition (with share CI) -> channelB.json
python3 dp4.py 200           # 4×4 exact DP anchor -> dp4.json
python3 plots_oracle.py      # survival/channelB/dp4.json -> figures/fig3,fig4,fig5

# 4×4 afterstate-FVI dual-gate certification (the only file depending on torch; the analysis pipeline stays zero-dependency)
python3 -m venv .venv && .venv/bin/pip install torch
.venv/bin/python rl4.py all  # mode-γ + mode-T training + dual gate -> rl4_gate.json
```

## Direction ① learned value function — 4×4 afterstate-FVI dual-gate certification (the third independent estimate of the online ceiling, step one)

The EVPI information share 57–69% leans on the assumption that "strong (beam, no lookahead) = the true online ceiling." Direction ① re-measures this independently with a **learned value function orthogonal to search**. Before burning 8×8 compute, we first prove on **4×4** (where exact-DP ground truth exists) that the afterstate fitted value iteration (FVI) pipeline is trustworthy — otherwise, when 8×8 fails to beat beam-strong, we can't tell "we've hit the ceiling" from "the net wasn't trained well" (the weak-agent trap). `rl4.py` trains two heads (MLP afterstate value networks):

| gate | criterion (pre-registered) | ground truth | result |
|---|---|---|---|
| **γ-gate value** | \|V_net(empty) − 3.9157\| < 0.05 | value_iteration(γ=0.95)=3.9157 | **PASS** Δ=0.0013 |
| **γ-gate policy** | greedy-on-V_net optimal-ratio paired-CRN bootstrap 95% CI lower bound ≥ 0.92 | myopic greedy 0.859 | **PASS** ratio 1.005, CI[0.997,1.013] |
| **T-gate value** | T∈{8,16} \|V_net(empty,T) − bdp_T(T)\|/bdp_T(T) < 5% | bdp_T(8)=3.5934, (16)=4.8637 | **PASS** 2.2% / 3.9% |
| **T-gate online** | M=50k undiscounted T-rollout relative deviation < 5% + precision guardrail 2·SE < 2.5%·bdp_T | bdp_T(T) | **PASS** 1.6% / 1.5%, guardrail passed |
| **displacement check** | probe-set V change > τ_disp (V-units, binding); corr(V_net,heuristic4)<0.92 (advisory) | corr(heuristic4,V\*)=0.870 | **PASS** disp 0.74>0.22; corr 0.871 |

**Both gates PASS ⇒ certifies the 4×4 afterstate-FVI pipeline** (value network + FVI loop + γ/undiscounted-T backups + ground-truth reproduction + displacement check). Results in `rl4_gate.json`.

**Not certified (must be stated)**: 4×4 is **single-piece/round + linear scoring (no combo)**. The dual gate does **not** certify the 8×8
beam_hand three-piece-hand candidate enumeration × afterstate ensemble — that layer is backstopped by a future 8×8 competence gate (vs beam-strong
paired-CRN + pre-registered effect size). This certification only says "the FVI machinery can reproduce the true optimum + true finite-T value on an exactly solvable small board."

Engineering notes: mode-γ uses hidden=256 (the tight 0.05 absolute-value gate needs to suppress the max-operator's overestimation bias — a small net overestimates by ~4%); mode-T uses hidden=128 (the 5% relative gate is looser, and each sweep is ~2.5× faster, converging in 102 sweeps / 637s).

## Files

| file | role |
|---|---|
| `pieces.py` | 38-piece catalog (rotation generation + deduplication) |
| `scoring.py` | non-linear scoring model (multi-clear/combo/all-clear, tunable) |
| `sim.py` | readable reference engine + variance decomposition |
| `fast.py` | bitboard high-speed engine + beam-strong + lookahead + beam rollout |
| `ladder.py` | skill ladder + crossover point |
| `compare.py` | paired head-to-head (CRN, verifying rollout regression) |
| `experiments.py` | variance/ladder/convergence -> `results.json` |
| **`oracle_analysis.py`** | **luck two channels (novel main line): seer/blind/anti players + fixed horizon + survival curves + EVPI decomposition + bootstrap CI** |
| **`dp4.py`** | **4×4 exact DP anchor: value iteration true optimum + offline DP + greedy, discounted VOI + heuristic gap + backward_dp_T undiscounted finite-T ground truth** |
| **`rl4.py`** | **direction ① 4×4 afterstate-FVI dual-gate certification (torch): mode-γ/mode-T value network + FVI + undiscounted T-rollout + displacement check -> `rl4_gate.json`** |
| `plots.py` | zero-dependency SVG (fig1 players / fig2 ladder) |
| `plots_oracle.py` | zero-dependency SVG (fig3 survival / fig4 EVPI / fig5 4×4 anchor) |

## License
MIT
