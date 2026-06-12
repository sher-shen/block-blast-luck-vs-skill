# Luck as the Value of Hindsight: Quantifying Luck vs. Skill in a Single-Player Stochastic Block-Puzzle

*Working draft — target venue: IEEE CoG / ACM FDG / FUN (opportunistic). Status: draft v1 (续20), assembled from the project's frozen results + two new robustness/design studies (A′, B).*

---

## Abstract

Single-player block-puzzle games (Block Blast, Tetris-likes, "1010!") are enormously popular yet
informally dismissed as "all luck" or "all skill." We make the question quantitative for an 8×8
Block Blast variant by operationalizing **luck = the value of hindsight**: the expected score a player
forgoes by *not* knowing the future piece stream. We decompose the gap between a perfect-foresight
oracle and the strongest no-foresight online policy into two channels — **survival luck** (the
probability of an unavoidable death) and **scoring luck** (the *expected value of perfect information*,
EVPI, conditional on survival) — estimated with per-seed paired bootstrap CIs under a fixed horizon,
and anchored by an **exact dynamic-programming solution on a tractable 4×4 board**. On 8×8, survival
luck is negligible (a perfect-foresight player almost never dies; per-round death hazard ≲ 3×10⁻⁴),
while scoring luck accounts for **57–69%** of the oracle's score advantage. Replacing the hand-crafted
online ceiling with a learned value-guided lookahead policy (our strongest playable agent, +25% over
the hand-crafted baseline) **raises** the luck share toward ~100%, confirming a pre-registered
prediction that the 57–69% figure is a conservative lower bound. We then contribute two studies. **(A)**
A 2×2 sensitivity analysis shows the luck share is robust to — and in fact *increases* under — a more
realistic benchmark (community-reverse-engineered multiplicative-combo scoring and benevolent piece
dealing), reaching **92–95%**; the original estimate is the lowest corner of the design space. **(B)**
Treating luck as a *design knob*, a forward dynamic-difficulty sweep finds that the piece-dealing
distribution controls **difficulty (survival) strongly and monotonically** (player survival 0.21→0.99)
but leaves the **luck share nearly flat (~65–75%)** — difficulty and luck-content are largely
**decoupled**, with luck-content governed instead by the *scoring structure*, not by how hard the
pieces are. We release a zero-dependency analysis pipeline and all figures.

---

## 1. Introduction

"Is this game luck or skill?" is usually argued, not measured. For *competitive* games there is a rich
literature (e.g., decomposing match outcomes into skill and chance components). For *single-player
stochastic survival* puzzles — where a randomly dealt stream of pieces must be packed onto a grid,
full lines clear, and the game ends when no piece fits — the question is rarely formalized at all.

We study an 8×8 Block Blast variant (38 distinct rotation-deduplicated pieces dealt three-at-a-time;
nonlinear scoring with multi-line and combo bonuses). Our framing is a single line of folklore made
precise:

> **Luck is a lack of information.** The luckiest possible player is one who *knows the future*.

So the **value of luck** is the **value of that information**: the score advantage of a
perfect-foresight oracle over the best policy that does *not* see the future. In decision-theoretic
terms this is the **expected value of perfect information (EVPI)**. Our contributions:

1. **A two-channel operationalization** of luck (survival vs. scoring/EVPI) with honest estimators
   (per-seed paired differences, bootstrap CIs, no ratio-of-means), §3.
2. **An exact 4×4 DP anchor** confirming that even a provably optimal online policy realizes only
   ~25% of the offline (hindsight) optimum — information value dominates, §4.1.
3. **A learned online ceiling** (value-guided lookahead, +25% over a strong hand-crafted baseline) that
   *raises* the measured luck share, confirming a pre-registered robustness check, §4.2.
4. **(New, A′) A benchmark-sensitivity study**: the luck share is a *conservative lower bound* —
   realistic scoring and dealing push it from 57–69% toward 92–95%, §5.
5. **(New, B) A dynamic-difficulty (DDA) forward study**: difficulty and luck-content **decouple** —
   dealing controls survival strongly but luck-share barely, §6.

A recurring methodological theme is **honesty about ill-posed estimands**: an earlier headline number
(a "78% gap") was *retracted* when we found the unbounded-horizon oracle is near-immortal and its score
diverges, making the ratio undefined; the fixed-horizon two-channel design (§3) was introduced
specifically to keep the estimand well-posed. The same discipline shaped this draft: a naive "re-run
luck/skill on an endless benchmark" reproduces exactly that pathology and was rejected in favor of the
well-posed A′ design (§5).

## 2. Related work & novelty

- **Tetris hardness/competitiveness** (Gehnen & Venier, *Tetris Is Not Competitive*, FUN 2024) studies
  worst-case adversarial structure; we instead measure *average-case information value* under a fixed
  dealing distribution. Different question, different machinery.
- **Skill/luck decomposition** literature targets competitive/multiplayer outcomes; we target a
  single-player stochastic-survival MDP and decompose on the *information* axis (EVPI) plus a *survival*
  axis, anchored by exact DP.
- **Strong play in piece-packing games**: 2048 (n-tuple networks + expectimax) and Tetris (CEM over
  hand features, e.g., Dellacherie) define the SOTA method families; we implement a CEM linear-feature
  baseline and a value-guided lookahead agent and use them as *online ceilings* for the luck analysis,
  not as ends in themselves.

**Gap we fill**: to our knowledge this is the first quantitative EVPI + survival decomposition of luck
vs. skill for a single-player stochastic-survival block-puzzle, anchored by an exact-DP information-value
ceiling, with an explicit benchmark-sensitivity and difficulty-vs-luck-decoupling analysis.

## 3. Method

**MDP.** State = (board occupancy, current 3-piece hand, combo counter). A *hand* is placed
piece-by-piece; completed rows/columns clear; consecutive clearing placements build a combo;
the episode ends when the current hand cannot be placed. Scoring (per `scoring.py`) is nonlinear:
per-cell points, a super-linear multi-line clear table, combo bonuses, and an all-clear bonus.

**Fixed horizon.** A perfect-foresight player is *near-immortal*; with an unbounded horizon its score
diverges and any "gap ratio" is undefined (the retracted-78% pathology). We therefore evaluate all
players over a **fixed horizon T** (T ∈ {40, 50, 60}) so scores are finite and comparable, and split
luck into:

- **Channel A — survival luck**: probability of death before T (a *hazard*, reported as a per-round
  rate with a one-sided Poisson upper bound, not a single-T death rate).
- **Channel B — scoring luck (EVPI)**: conditional on survival, the score advantage attributable to
  *information*, isolated by three structurally identical players differing only in their information set:
  - `strong` — strongest no-foresight online policy (beam search over the hand, no lookahead);
  - `blind` — a D-step lookahead player whose rollouts average over **S sampled** futures (correct
    marginalization over the *known* dealing distribution, but no real information);
  - `seer` — the same lookahead player whose rollout uses **the one true** future (perfect D-step
    foresight; a realizable-foresight *lower bound* on the true optimum).

  Then, on the cohort where strong ∧ blind ∧ seer all survive to T (so the difference is pure score),
  ``raw(seer − strong) = EVPI(seer − blind) + procedure(blind − strong)`` — **unclipped**. The luck
  share is the **per-seed paired median** of EVPI/raw (reporting the n with positive denominator and the
  conditional-on-seer>strong caveat), with bootstrap 95% CIs throughout. Common random numbers (a shared
  deal stream per seed) reduce paired variance.

## 4. Core results (8×8)

**4.1 Exact 4×4 DP anchor.** The full game is intractable to solve exactly, but a 4×4 board is not
(`dp4.py`, value iteration). There, a *provably optimal* online policy realizes only **≈25%** of the
offline (perfect-hindsight) DP optimum — i.e., **information value dominates** even under optimal online
play. This is an analogy (1 piece/round, linear scoring, γ-discounted), not a calibration, but it
independently confirms the qualitative headline on a board where "optimal" is not in doubt.

**4.2 Two channels on 8×8.**

- **Channel A (survival):** the seer is near-immortal — per-round death hazard point estimate
  ~7×10⁻⁵, one-sided 95% Poisson upper bound ~3.3×10⁻⁴/round. Death is essentially an avoidable
  technical failure (skill), not irreducible luck, save for rare unavoidable killer sequences.
- **Channel B (scoring/EVPI):** with `strong` as the online ceiling, the luck share is
  **69% / 65% / 57%** at T = 40 / 50 / 60 (cohort n = 54 / 41 / 34, per-seed paired median, bootstrap CIs).

**4.3 Stronger online ceiling raises the luck share (pre-registered).** The 57–69% loads on the
assumption "`strong` = the true online ceiling." A learned **value-guided lookahead** agent (trained
afterstate value + shallow search) scores **2924 vs. 2364 (+25%)** over `strong` and is our strongest
playable, no-foresight policy. Substituting it as the denominator on a frozen all-survivors cohort, the
luck share rises to **~100% (109/106/96%)** — exactly the pre-registered direction (a stronger online
player is a better marginalizer, so the residual to the oracle becomes almost pure information value).
The strong-denominator share reproduces the channel-B numbers on the same cohort (self-consistency).
Shares >100% are a *signal* (the learned player out-marginalizes the sampled-future `blind`), not a bug.

## 5. (New, A′) The luck share is a conservative lower bound

**Question.** The 57–69% headline was measured under a specific benchmark: a fixed T=50-style cap,
a *made-up* ("assumed") scoring table, and *uniform* piece dealing. A separate study found this
benchmark is artificially harsh — uniform 38-piece dealing and the horizon cap each compress survival
and score by 2–3×. Does the *luck share* survive a more realistic benchmark?

**Design (well-posed).** We hold the horizon **fixed** (keeping EVPI well-posed — a naive endless-horizon
re-run reproduces the retracted pathology) and vary only two well-posed knobs in a 2×2:
- **scoring**: `assumed` vs. `real_approx` (community-reverse-engineered multiplicative-combo scoring);
- **dealing**: `uniform` vs. `benevolent` (drop the uniquely board-wrecking 3×3 piece).

All three oracle players read the scoring rule and consume the deal stream, so each **re-optimizes
automatically** under any condition; the benevolent dealing distribution is threaded through the
internal rollout samplers too (so `blind`'s marginalizer stays correctly specified). The
`(assumed, uniform)` cell reproduces the channel-B numbers **exactly** (69/65/57% at n=54/41/34) — a
self-consistency check on the new code. (`oracle_realistic.py`, N=120, D=3.)

**Result.** The luck share moves **up** under both knobs:

| condition | T=40 | T=50 | T=60 |
|---|---|---|---|
| assumed · uniform (original) | **69%** | **65%** | **57%** |
| assumed · benevolent | 74% | 77% | 75% |
| real_approx · uniform | 87% | 86% | 82% |
| real_approx · benevolent (most realistic) | **95%** | **92%** | **93%** |

Under `real_approx`, the absolute oracle gap compresses sharply (`strong` and `seer` medians ~1408 vs.
1565 at T=50, a ~11% gap vs. ~62% under `assumed`) and `blind` even out-survives `strong` — yet the
*luck share* of that smaller gap rises to ~86%. **The original 57–69% is the lowest corner of the
design space**; a realistic benchmark only strengthens the conclusion that, at the playable ceiling,
the residual gap to hindsight is almost entirely irreducible luck. (Figure: `figures/evpi_realistic.svg`.)

**Honest limits.** `real_approx` is community-sourced and non-authoritative; the learned +25% agent is
*not* used as the denominator here because its value head is trained on `assumed`+`uniform` and would be
out-of-distribution under `real_approx` — we keep A′ to the auto-re-optimizing analytic players and
flag the learned-policy realistic re-test as future work.

## 6. (New, B) Difficulty and luck-content are decoupled

**Question.** The dealing distribution is the one knob a designer can freely turn (scoring and board
size are usually fixed). Treating luck as a *designable quantity* (the inverse of dynamic-difficulty
adjustment, DDA), how does the luck share respond as we slide a single difficulty knob?

**Design (forward only).** A one-parameter dealing family `w_i ∝ exp(−β·size_i)`: β=0 is uniform, β>0
favors small pieces (benevolent/easy), β<0 favors large (adversarial/hard). We sweep β and measure
survival + luck share at fixed T=50, assumed scoring (`dda_forward.py`, N=100). The full inverse problem
(solve for a dealing distribution hitting a target luck share) is deliberately *not* attempted; instead
a target is read off the forward curve by interpolation.

**Result (decoupling).** Difficulty is cleanly controllable; luck-content is not:

| β | E[piece size] | strong survival@T | luck share |
|---|---|---|---|
| −0.20 (adversarial) | 4.25 | 0.21 | 73% |
| 0.00 (uniform) | 3.82 | 0.76 | 65% |
| +0.40 (benevolent) | 3.24 | 0.99 | 67% |

Survival swings **0.21 → 0.99 monotonically** with β, but the luck share stays **~65–75% (CIs overlap)**
across the entire range. So a designer can dial *difficulty* (survival) precisely via dealing **without
changing how much of the outcome is luck**. Combined with A′, the luck-content is set by the **scoring
structure** (multiplicative combo: 65%→86%) and by the presence of *specific* extreme blockers (removing
the 3×3: 65%→77%) — **not** by how big the dealt pieces are on average. (Figure: `figures/dda_forward.svg`.)

## 7. Honest limitations

- **EVPI is "≈" not "=":** `seer` previews only D steps (saturated at D≥3) — a *lower bound* on the true
  offline optimum; `blind` approximates distributional-optimal marginalization with S samples (Monte
  Carlo noise). The two biases point opposite ways, so we report magnitude + CIs and lean on the
  observed flattening in S and D.
- **The 4×4 DP is an analogy, not a calibration** (1 piece/round, linear scoring, γ-discount); it
  confirms direction, not the exact 8×8 number.
- **Scoring & dealing are models.** `assumed` is explicitly made-up; `real_approx` is community-sourced
  and inconsistent across sources. The real app's mechanism is unpublished. A′ is therefore a
  *sensitivity* analysis, not a claim of reproducing the commercial game.
- **Benevolent dealing is a lower bound on kindness** (static, board-state-independent reweighting; no
  "guarantee at least one placeable piece" mechanic, which would couple to the policy and break CRN).
- **Learned-policy realistic re-test deferred** (§5 limits): the +25% agent under `real_approx`/benevolent
  is OOD and left to future work.

## 8. Reproducibility

Zero-dependency analysis pipeline (Python stdlib only; the single learned-value file isolates Torch).
Key entry points:
```
python oracle_analysis.py channel 120 3 40,50,60   # core two-channel EVPI -> channelB.json
python dp4.py 200                                   # exact 4×4 DP anchor -> dp4.json
python evpi_vla.py 120                              # learned-ceiling luck share -> evpi_vla.json
python oracle_realistic.py 120                      # (A′) 2x2 benchmark sensitivity -> evpi_realistic.json
python dda_forward.py 100                           # (B)  forward DDA sweep -> dda_forward.json
```
Figures: `figures/{evpi_vla,evpi_realistic,dda_forward,endless_survival,deal_audit,calibrate_scoring}.svg`.

## 9. Conclusion

Luck in a single-player stochastic block-puzzle is **the value of hindsight**, and it is large: even a
provably optimal online policy (4×4) leaves ~75% on the table to a perfect-foresight oracle, and on 8×8
the residual gap to hindsight is 57–69% of the oracle's advantage under a conservative benchmark — rising
to ~100% against the strongest learned online policy and ~92–95% under realistic scoring and dealing.
Difficulty and luck-content are separable design axes: the dealing distribution sets *how hard* the game
is, but *how much of it is luck* is fixed by the scoring structure. The methodological throughline —
retract ill-posed estimands, pre-register the robustness direction, anchor with exact DP — is the part we
would most like to see reused.
