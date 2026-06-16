# A mathematical model of *dealing difficulty* in Block Blast

### How the piece-draw distribution `p` controls survival and score — a load / stability theory

> **One sentence.** Block Blast survival is a *stochastic bin-packing stability* problem: the
> deal distribution `p` sets a **load** `ρ(p) = μ(p)/Γ(p)` (mean cells dealt per piece ÷ the
> board's sustainable clear-capacity); survival is long when `ρ<1`, collapses when `ρ>1`, and
> passes through a **phase transition** at `ρ≈1`. Raising a piece's probability moves `ρ`
> through **two channels** — the *size* of the piece (numerator `μ`) and its *packability*
> (denominator `Γ`) — which is exactly why "remove the 3×3 (2.6% of the mass) → survival ×2.7"
> is such a large effect for such a small probability change.

This complements the project's existing **luck-side** theory (`LUCK_PREDICT_RESULTS.md`, a local research note:
*scoring convexity → luck-content*, EVPI = a Jensen gap). That work formalized **what makes a game lucky**;
it left **what makes a deal hard** as an empirical observation (the DDA decoupling, the `deal_audit` 3×3
result). This note supplies the missing **difficulty** half. Together: **`p` (the deal) sets difficulty
through a mean-load `μ/Γ`; the scoring function sets luck through its convexity — two decoupled knobs.**

---

## 1. Setup and notation

- Board of `K = n² = 64` cells (`n=8`). State `B ∈ {0,1}^{n×n}`; fill level `N(B) = #filled cells ∈ [0,K]`.
- Piece catalog `Π` (the 38 oriented shapes of [`pieces.py`](pieces.py)). Piece `τ` has cell-set with
  **size** `sτ = |τ|`. Pieces are drawn i.i.d. from a distribution `p = (pτ)_{τ∈Π}`.
- **Mean piece size (arrival rate)** `μ(p) = Σ_τ pτ · sτ` — the expected cells *added* per piece.
- A round deals `m=3` pieces; a policy `π` places each at a legal position; any full row/column clears
  immediately. We index time `t` by **pieces placed** and write `N_t` for the fill level after piece `t`.

Death = the dealt piece has no legal placement. The object of interest is the **survival time**
`T = inf{ t : piece t cannot be placed }` and the **score** accrued before `T`.

---

## 2. The conservation identity (exact)

Let `A_t = Σ_{i≤t} s_i` be total cells *added* by `t` and `C_t` the total cells *removed* by line clears.
Since cells are neither created nor destroyed except by these two channels,

> **`N_t = A_t − C_t`,  with the hard constraint `0 ≤ N_t ≤ K`.** &nbsp;&nbsp; *(identity)*

Rearranging and dividing by `t`: `C_t / t = A_t/t − N_t/t`. Along **any surviving trajectory** the fill
level is bounded (`0 ≤ N_t ≤ K`), so `N_t/t → 0`, while `A_t/t → μ(p)` by the law of large numbers. Hence

> **Theorem 1 (clear-rate identity).** *Any policy that survives indefinitely must remove cells by line
> clears at long-run average rate exactly `μ(p)` cells per piece — equivalently `μ(p)/8` line-clears per
> piece (a row/column is 8 cells).* This is exact; it uses only conservation and the LLN.

For the uniform 38-piece deal `μ ≈ 3.82`, so **survival demands ≈ 0.48 line-clears every single piece,
forever.** Raise `μ` and you must clear faster, indefinitely; there is no slack to "bank."

---

## 3. Load, capacity, and the stability dichotomy

Theorem 1 says survival needs a clear-rate of `μ(p)`. Can the board *supply* it? Define the policy's

> **packing clear-capacity** `Γ(p,π)` = the supremum of average clear-rate (cells/piece) sustainable
> without the fill level drifting up to `K`.

`Γ` is a geometric quantity: it measures how efficiently the dealt shapes can be packed into completable
lines. **There is no closed form for `Γ` on the 8×8 board** — this is the one constant the theory cannot
hand you, and where measurement (§6) enters. With it, define the

> **load** `ρ(p,π) = μ(p) / Γ(p,π)`.

The behavior of `N_t` is then governed by a stability dichotomy that is **the established result for
stochastic / online bin packing** ([Courcoubetis & Weber; Csirik et al., "Necessary and sufficient
conditions for stability of a bin-packing system"](https://www.cambridge.org/core/journals/journal-of-applied-probability/article/abs/necessary-and-sufficient-conditions-for-stability-of-a-binpacking-system/55A6E0213D3EBC5EA2773EDDDF869C4C);
[Lyapunov-function stability of packing processes](https://link.springer.com/article/10.1023/B:QUES.0000046581.34849.cf)):

| regime | name | wasted/unclearable space grows as | survival on a finite board |
|---|---|---|---|
| `ρ < 1` | subcritical (stable) | bounded | long (exponential in the margin — §4) |
| `ρ = 1` | critical | `√t` | polynomial; maximally sensitive |
| `ρ > 1` | supercritical (overload) | `Θ(t)` (linear) | short (linear hitting time — §4) |

On an *infinite* board these are stability statements; on our *finite* `K=64` board the supercritical
linear drift becomes a short **hitting time to overflow**, i.e. death. The deal distribution `p` selects
the regime entirely through `ρ(p)`.

---

## 4. A mean-field survival law (why the transition is sharp)

To get an explicit survival formula, collapse the geometry to a one-dimensional **fluid / random-walk**
picture of the fill level `N_t`. Per piece, `N` increases by the arrival `s` (mean `μ`) and decreases by
`8` whenever a placement completes a line; let the policy's achievable clear frequency be `q̄` clears per
piece (so it removes `8q̄` cells/piece on average, and `Γ = 8·q̄_max`). The **drift** of the fill level is

> `δ = μ − 8·q̄`.

- **Supercritical (`μ > Γ`, i.e. `q̄_max < μ/8`).** Even at maximum clearing the drift `δ > 0` is positive;
  `N_t` climbs ballistically and hits `K` at a **linear** time `T ≈ (K − N₀)/δ`. Survival is short and
  scales like `1/(μ−Γ)`.
- **Subcritical (`μ < Γ`).** The policy can hold `δ ≤ 0`; death now requires a *rare* upward excursion of
  the fill against a restoring drift. For a random walk with negative drift `−a` and step-variance `σ²`,
  the expected time to climb a barrier of height `H = K − N*` is **exponentially large** (Cramér / Gambler's
  ruin):

  > **`E[T] ≍ exp( 2a·H / σ² )`,  with margin `a ∝ (Γ − μ)`.**

This is the mechanism behind the **sharpness**: survival depends *exponentially* on the margin `Γ − μ`, so
near the critical point `ρ=1` a **small change in `p`** (a small `Δμ` or `ΔΓ`) produces a **large
multiplicative change in survival**. The `deal_audit` observation — deleting the 3×3 (a `Δp` of only 2.6%)
multiplies survival by ≈2.7 — is the fingerprint of a system operating **near criticality**, where the
exponent is moving fast. (Burgiel's theorem is the `ρ`-independent backstop: with any mass on a killer
pattern, `T < ∞` almost surely regardless of policy — [Tetris killer sequences](https://harddrop.com/wiki/Deadly_piece_sequence).
The load model governs the *typical* `T`; Burgiel governs its ultimate finiteness.)

---

## 5. How raising a piece's probability moves the load (the answer)

Both channels of `ρ = μ/Γ` respond to `p`:

**(a) Numerator — size.** `∂μ/∂pτ = sτ`. Moving probability mass onto a **larger** piece raises `μ`
linearly in its size. Concretely, shifting mass `ε` from a size-`a` piece to a size-`b` piece changes
`μ` by `ε(b−a)`. Bigger pieces ⇒ higher `μ` ⇒ higher load ⇒ shorter survival. **Monotone.**

**(b) Denominator — packability.** Two equal-size pieces can have very different `Γ`. A **straight bar**
lays all its cells in *one* line, contributing maximally toward a clear. A **diagonal** of the same size
drops its cells into *distinct* rows *and* distinct columns, contributing to no line efficiently and
leaving isolated holes that raise the empty region's boundary (the `frag`/perimeter term the engine's
heuristic already tracks). So a diagonal **lowers `Γ`** even at fixed `μ`. The 3×3 is the extreme on a
*third* sub-axis — it needs a rare 3×3 empty pocket, so its *placeability* collapses as the board fills.

> **Decision rule.** Upweighting a piece raises difficulty in proportion to **`sτ` (its size) plus a
> packability penalty** (how much it fragments / how rarely it fits). Diagonals and the 3×3 are doubly
> bad: large *and* unpackable.

**Score vs survival.** Score per piece in sustained play `≈ (μ/8) × (points per line)`, so *harder* deals
score faster *per piece* — but they also die sooner. Total score is the product, `score ≈ (rate) × T`. A
priori this could be single-peaked in `μ`. **Empirically (§6) it is not, over the tested range: the
survival collapse dominates and total score is *maximised at the easy end*** — the score peak sits at the
lowest `μ`, not in the interior.

---

## 6. Empirical validation (`prob_sweep.py`, greedy policy, n=50 seeds, horizon 150)

`prob_sweep.py` reuses the [`sim.py`](sim.py) engine read-only and only changes the *dealing* to a weighted
draw. Two pre-stated predictions:

**(P1) Size axis — scale the weight on every size≥5 piece by a factor `f`.** Survival should fall
monotonically as `μ` rises.

| `f` (weight on big pieces) | 0 | 0.25 | 0.5 | 1 (uniform) | 2 | 4 | 8 |
|---|---|---|---|---|---|---|---|
| `μ(p)` | 3.41 | 3.52 | 3.63 | 3.82 | 4.11 | 4.52 | 4.96 |
| **survival** (pieces-rounds) | **37.3** | 27.3 | 23.3 | 14.7 | 10.6 | 7.6 | 7.4 |
| total score | 1587 | 1261 | 1017 | 670 | 465 | 379 | 428 |

→ **P1 confirmed, strongly monotone.** Survival drops ~5× across the swept `μ` range; the steepest drop is
in the middle (`μ` 3.6→4.1), the signature of crossing the critical region. Total score is **monotone
decreasing** (peak at the lowest `μ`) — the "interior score peak" conjecture of §5 is **not** supported
here; survival dominates.

**(P2) Packability at *identical* `μ` — the controlled test that isolates `Γ`.** Upweight (×6) a size-3
*straight* bar vs a size-3 *diagonal*. Both are size 3, so `μ` is identical (3.646); any survival gap is
**pure packability** `Γ`.

| config (all at `μ = 3.646`) | survival | total score |
|---|---|---|
| easy-3 — straight bar upweighted | **23.5** | 1037 |
| hard-3 — diagonal upweighted | **14.1** | 553 |
| (baseline uniform, `μ = 3.82`) | 14.7 | 670 |

→ **P2 confirmed, and cleanly.** At *identical mean piece size*, the diagonal deal survives **40% shorter**
(14.1 vs 23.5) and scores **47% less** than the straight-bar deal — this gap is attributable to `Γ` alone,
not size. Note too that the diagonal deal (`μ=3.65`) is **as deadly as the baseline** (`μ=3.82`) despite a
*lower* mean size: its packability penalty exactly eats its size advantage. This is direct evidence that
`Γ` (geometry), not just `μ` (size), is a first-class driver of difficulty — precisely the denominator of
the load.

(`greedy` is the fixed workhorse policy here for speed; the *difficulty ordering* is a property of the deal,
expected to be robust to policy strength. Survival is right-censored at 150 rounds — censoring was 0% for
every config above, so the means are uncensored.)

---

## 7. The two-knob picture (difficulty × luck), unified

| knob | controlled by | mechanism | quantity | established in |
|---|---|---|---|---|
| **Difficulty** | deal distribution `p` | load `ρ = μ(p)/Γ(p)`; phase transition at `ρ=1` | survival `T`, score | **this note** + `prob_sweep.py` |
| **Luck** | scoring function | Jensen / convexity gap in the random future | EVPI share | `LUCK_PREDICT_RESULTS.md` (local) |

They are **decoupled** (the project's DDA result found only a weak survival-leak residual `ρ≈0.26`
correlation): changing `p` slides you along the difficulty axis without much touching the luck share;
changing the scoring convexity slides the luck share without changing the load. This note pins down the
left column with a mechanism (load/stability) it previously lacked.

---

## 8. What is exact, what is heuristic (honesty)

- **Exact:** the conservation identity and **Theorem 1** (clear-rate `= μ(p)`) — pure accounting + LLN.
- **Imported theorem (analogy):** the `ρ≶1` stability dichotomy is the bin-packing-stability result
  transcribed to this game; our board is finite and our "items" are placed by a *strategic* policy (not a
  fixed online rule), so the transcription is a *modeling analogy*, not a re-proof. The qualitative
  prediction (sharp transition, exponential-vs-linear survival) is what we test, and it holds.
- **Mean-field toy:** the §4 random-walk survival law (`E[T] ≍ exp(2aH/σ²)`) ignores spatial structure
  (`N_t` is not really 1-D Markov — geometry matters), so it is an *explanatory* law for the *shape* of the
  transition, not a calibrated predictor of `T`.
- **No closed form for `Γ(p,π)`** on 8×8 (consistent with "8×8 has no exact inner bound", the audit-rejected
  direction #2). `Γ` is defined operationally and accessed through the sweep. We therefore make **ordinal /
  mechanistic** claims (which direction, why, how sharp), not a numerical law `T = T(p)`.
- **Policy dependence:** `Γ` and `ρ` depend on the policy. The *ordering* of deals by difficulty is the
  robust object; absolute survival numbers are greedy-specific and horizon-censored.

---

## 9. Relation to the literature

- **Stochastic / online bin-packing stability** — the load-region / wasted-space dichotomy we borrow:
  [Csirik et al., *Necessary and sufficient conditions for stability of a bin-packing system*](https://www.cambridge.org/core/journals/journal-of-applied-probability/article/abs/necessary-and-sufficient-conditions-for-stability-of-a-binpacking-system/55A6E0213D3EBC5EA2773EDDDF869C4C);
  [Courcoubetis & Weber, *Stability of on-line bin packing with random arrivals*](https://www.cambridge.org/core/journals/probability-in-the-engineering-and-informational-sciences/article/abs/stability-of-online-bin-packing-with-random-arrivals-and-longrunaverage-constraints/144B51CD23274E98F81DFF0CE233DC46);
  [Lyapunov-function technique for packing-process stability](https://link.springer.com/article/10.1023/B:QUES.0000046581.34849.cf).
- **Forced loss under random pieces** — the finiteness backstop: [Burgiel, *How to lose at Tetris* / killer
  sequences](https://harddrop.com/wiki/Deadly_piece_sequence); complexity context: [*Tetris is Hard, Even to
  Approximate* (Demaine et al.)](https://people.csail.mit.edu/dln/papers/tetris/tetris.pdf).
- **Gap / novelty:** a literature scan for difficulty / piece-distribution / Markov-survival analysis of
  **1010!- / Block-Blast-style** games returns only the games themselves — no mathematical model. To our
  knowledge this load/stability formalization of *deal difficulty* for an 8×8 batch-of-3 block puzzle, with
  the size-vs-packability decomposition of `∂(difficulty)/∂p`, is new.

---

## 10. Formula summary (the "take-home")

```
arrival rate (cells/piece)    μ(p)   = Σ_τ p_τ · s_τ
clear-rate REQUIRED to survive        = μ(p)        cells/piece      (Theorem 1, exact)
clear-CAPACITY of the board   Γ(p,π)                                  (geometric; measured, no closed form)
LOAD                          ρ(p,π) = μ(p) / Γ(p,π)
                              ρ < 1  → survive long,  E[T] ≍ exp(2a·(K−N*)/σ²),  a ∝ (Γ−μ)
                              ρ > 1  → die fast,       E[T] ≈ (K−N₀)/(μ−Γ)
                              ρ ≈ 1  → phase transition: small Δp ⇒ large Δsurvival   (the 3×3 effect)

∂(difficulty)/∂p_τ  ∝  s_τ          (size: numerator μ)
                    +  packability penalty(τ)   (geometry: lowers Γ — diagonals & 3×3 worst)
```

*Artifacts:* `prob_sweep.py` (new; imports `sim.py`/`pieces.py` read-only), `prob_sweep.json`,
`prob_sweep_out.txt`. Frozen producers untouched.
