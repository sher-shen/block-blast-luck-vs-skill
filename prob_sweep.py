"""
prob_sweep.py — empirical validation of the DEAL-DISTRIBUTION -> DIFFICULTY model
(see DIFFICULTY_MODEL.md). NEW file; imports sim.py / pieces.py READ-ONLY (firewall-safe).

Question: as we change the piece-draw probabilities p, how do survival and score change,
and WHY? The model (DIFFICULTY_MODEL.md) predicts difficulty is governed by the LOAD
    rho(p) = mu(p) / Gamma(p),     mu(p) = E[piece size],  Gamma = packing clear-capacity.
Two testable predictions checked here:
  (P1) SIZE axis: survival is monotone DECREASING in mu(p) (upweighting big pieces raises
       the numerator of the load). Total score may be SINGLE-PEAKED in mu (need big pieces
       to score via multi-clears, but too many kill you).
  (P2) PACKABILITY axis (the clean controlled test): at IDENTICAL mu, upweighting a
       hard-to-pack size-3 piece (diag3, a diagonal) gives SHORTER survival than upweighting
       an easy size-3 piece (tromino_I, a straight bar). Same mu -> any gap is pure Gamma.

Engine reused read-only from sim.py: empty_board / policy_greedy / SHAPES / NUM_TYPES.
We only change the DEALING (weighted instead of uniform); the placement engine is untouched.

Usage:  python prob_sweep.py [n_seeds] [max_rounds]
"""

import sys, json, random
from statistics import mean, pstdev
from pieces import CATALOG
from sim import empty_board, policy_greedy, SHAPES, NUM_TYPES

NAMES = [name for name, _ in CATALOG]
SIZES = [len(cells) for cells in SHAPES]
assert len(NAMES) == NUM_TYPES == len(SHAPES)


def mu_of(weights):
    """Mean piece size E[s] under draw distribution proportional to `weights`."""
    W = sum(weights)
    return sum(w * s for w, s in zip(weights, SIZES)) / W


def idxs_with_name_prefix(prefix):
    return [i for i, nm in enumerate(NAMES) if nm.split('#')[0] == prefix]


def play_weighted(weights, seed, max_rounds, policy=policy_greedy):
    """One game with a weighted deal. Returns (rounds_survived, total_score, censored?)."""
    deal_rng = random.Random(f"deal-{seed}")
    act_rng = random.Random(f"act-{seed}")
    board = empty_board()
    combo = 0
    total = 0
    rounds = 0
    pool = list(range(NUM_TYPES))
    for _ in range(max_rounds):
        idxs = deal_rng.choices(pool, weights=weights, k=3)
        hand = [SHAPES[i] for i in idxs]
        pts, board, combo, alive = policy(board, hand, combo, act_rng)
        total += pts
        if not alive:
            return rounds, total, False
        rounds += 1
    return rounds, total, True   # censored = hit max_rounds alive


def eval_config(weights, n_seeds, max_rounds):
    surv, score, cens = [], [], 0
    for s in range(n_seeds):
        r, t, c = play_weighted(weights, s, max_rounds)
        surv.append(r); score.append(t); cens += 1 if c else 0
    return {
        "mu": round(mu_of(weights), 4),
        "surv_mean": round(mean(surv), 2), "surv_sd": round(pstdev(surv), 2),
        "score_mean": round(mean(score), 1), "score_sd": round(pstdev(score), 1),
        "censored_frac": round(cens / n_seeds, 3),
        "n": n_seeds, "max_rounds": max_rounds,
    }


def run(n_seeds=30, max_rounds=120):
    base = [1.0] * NUM_TYPES
    big = [i for i, s in enumerate(SIZES) if s >= 5]          # 5,6,9-cell awkward/large pieces
    tri_straight = idxs_with_name_prefix("tromino_I")          # size-3 straight bar (easy)
    tri_diag = idxs_with_name_prefix("diag3")                  # size-3 diagonal (hard to pack)

    results = {"meta": {"n_seeds": n_seeds, "max_rounds": max_rounds,
                        "num_types": NUM_TYPES, "uniform_mu": round(mu_of(base), 4),
                        "big_pieces": [NAMES[i] for i in big],
                        "policy": "greedy"},
               "size_sweep": [], "packability": {}}

    # ---- (P1) SIZE axis: scale the weight on all size>=5 pieces by factor f ----
    print(f"# prob_sweep  n_seeds={n_seeds} max_rounds={max_rounds} (greedy)\n")
    print("## (P1) SIZE axis — scale weight on size>=5 pieces by f")
    print(f"{'f':>6} {'mu':>7} {'surv':>8} {'score':>9} {'cens%':>6}")
    for f in [0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0]:
        w = base[:]
        for i in big:
            w[i] = f
        r = eval_config(w, n_seeds, max_rounds)
        r["f"] = f
        results["size_sweep"].append(r)
        print(f"{f:>6} {r['mu']:>7} {r['surv_mean']:>8} {r['score_mean']:>9} {r['censored_frac']*100:>5.0f}")

    # ---- (P2) PACKABILITY at fixed mu: upweight straight-3 vs diagonal-3 by same factor ----
    print("\n## (P2) PACKABILITY axis — upweight a size-3 piece by F=6 (same mu either way)")
    print(f"{'config':>14} {'mu':>7} {'surv':>8} {'score':>9} {'cens%':>6}")
    F = 6.0
    for label, idxs in [("baseline", []), ("easy-3 (straight)", tri_straight),
                        ("hard-3 (diagonal)", tri_diag)]:
        w = base[:]
        for i in idxs:
            w[i] = F
        r = eval_config(w, n_seeds, max_rounds)
        r["upweighted"] = [NAMES[i] for i in idxs]
        results["packability"][label] = r
        print(f"{label:>14} {r['mu']:>7} {r['surv_mean']:>8} {r['score_mean']:>9} {r['censored_frac']*100:>5.0f}")

    with open("prob_sweep.json", "w") as fp:
        json.dump(results, fp, indent=2)
    print("\n-> prob_sweep.json")
    return results


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    mr = int(sys.argv[2]) if len(sys.argv) > 2 else 120
    run(n, mr)
