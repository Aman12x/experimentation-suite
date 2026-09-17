#!/usr/bin/env python3
"""
Simulation report: does each method keep the guarantee it advertises?

Every row simulates data where the truth is known and measures the long-run
behaviour. The same properties are enforced (with tolerances) in tests/.

    python validate.py
"""

import warnings

import numpy as np
import pandas as pd

from modules import ABTestingEngine, CausalInferenceLab
from modules.experiment_design import to_unit_level

warnings.filterwarnings("ignore")
engine, lab = ABTestingEngine(), CausalInferenceLab()
rows = []


def record(check, target, measured, baseline=""):
    rows.append({'Check': check, 'Target': target, 'Measured': measured, 'Naive alternative': baseline})


# 1. False-positive rate under unequal variance and unequal group size
rng = np.random.default_rng(3)
sims = 2000
welch = student = 0
for _ in range(sims):
    c, t = rng.normal(0, 1, 1000), rng.normal(0, 4, 100)
    welch += engine.t_test(c, t)['significant']
    student += engine.t_test(c, t, equal_var=True)['significant']
record("t-test false-positive rate (unequal n and variance)", "5%",
       f"{welch / sims:.1%}", f"Student's pooled: {student / sims:.1%}")

# 2. Coverage of the relative-lift interval
rng = np.random.default_rng(0)
delta = naive = 0
for _ in range(sims):
    r = engine.t_test(rng.exponential(10, 150), rng.exponential(11, 150))
    delta += r['lift_ci_lower'] <= 10 <= r['lift_ci_upper']
    naive += r['ci_lower'] / r['control_mean'] * 100 <= 10 <= r['ci_upper'] / r['control_mean'] * 100
record("Relative-lift 95% CI coverage", "95%",
       f"{delta / sims:.1%}", f"absolute CI / control mean: {naive / sims:.1%}")

# 3. CUPED power at the same sample size
rng = np.random.default_rng(2)
sims_cuped = 500
plain = cuped = 0
reductions = []
for _ in range(sims_cuped):
    pre_c, pre_t = rng.normal(100, 20, 400), rng.normal(100, 20, 400)
    c = 0.9 * pre_c + rng.normal(0, 8, 400)
    t = 0.9 * pre_t + rng.normal(0, 8, 400) + 1.5
    r = engine.cuped_test(c, t, pre_c, pre_t)
    cuped += r['significant']
    plain += r['unadjusted_p_value'] < 0.05
    reductions.append(r['variance_reduction_pct'])
record("CUPED power to detect a true +1.5 effect (n=400/arm)", "higher is better",
       f"{cuped / sims_cuped:.1%} (variance cut {np.mean(reductions):.0f}%)",
       f"plain Welch: {plain / sims_cuped:.1%}")

# 4. Family-wise error with five null variants
rng = np.random.default_rng(8)
sims_mv = 600
raw = holm = 0
for _ in range(sims_mv):
    r = engine.multi_variant_test({f'v{i}': rng.normal(0, 1, 200) for i in range(6)}, 'v0')
    raw += any(c['significant_raw'] for c in r['comparisons'])
    holm += any(c['significant'] for c in r['comparisons'])
record("Chance of any false win, 5 null variants", "≤ 5%",
       f"{holm / sims_mv:.1%}", f"uncorrected: {raw / sims_mv:.1%}")

# 5. Peeking
rng = np.random.default_rng(10)
sims_seq = 1000
valid = peek = 0
for _ in range(sims_seq):
    r = engine.sequential_test(rng.normal(0, 1, 2000), rng.normal(0, 1, 2000), n_looks=20)
    valid += r['could_stop_at_look'] is not None
    peek += r['naive_peeking_would_stop_at_look'] is not None
record("False-positive rate with 20 interim looks", "≤ 5%",
       f"{valid / sims_seq:.1%}", f"peeking at fixed-horizon p: {peek / sims_seq:.1%}")

# 6. 2SLS interval coverage under confounding
rng = np.random.default_rng(3)
sims_iv = 400
covered = ols_covered = 0
for _ in range(sims_iv):
    n = 800
    z, u, x1 = rng.normal(size=n), rng.normal(size=n), rng.normal(size=n)
    t = 0.5 * z + 2 * x1 + u + rng.normal(size=n)
    y = 2.0 * t + x1 + 3 * u + rng.normal(size=n)
    r = lab.instrumental_variables(pd.DataFrame({'y': y, 't': t, 'z': z, 'x1': x1}), 'y', 't', 'z', ['x1'])
    covered += r['ci_lower'] <= 2.0 <= r['ci_upper']
    ols_covered += abs(r['ols_estimate'] - 2.0) < 0.1
record("2SLS 95% CI coverage with an unobserved confounder", "95%",
       f"{covered / sims_iv:.1%}", f"OLS within ±0.1 of truth: {ols_covered / sims_iv:.1%}")

# 7. Analysing at the unit of randomization
rng = np.random.default_rng(3)
sims_unit = 400
row_level = unit_level = 0
for _ in range(sims_unit):
    arm = rng.permutation(np.repeat(['control', 'treatment'], 150))
    level = rng.normal(50, 15, 300)
    sessions = rng.integers(1, 12, 300)
    df = pd.DataFrame({
        'user_id': np.repeat(np.arange(300), sessions),
        'variant': np.repeat(arm, sessions),
        'spend': np.repeat(level, sessions) + rng.normal(0, 3, sessions.sum()),
    })
    row_level += engine.t_test(df[df.variant == 'control'].spend.values,
                               df[df.variant == 'treatment'].spend.values)['significant']
    units, _ = to_unit_level(df, 'user_id', 'variant', ['spend'])
    unit_level += engine.t_test(units[units.variant == 'control'].spend.values,
                                units[units.variant == 'treatment'].spend.values)['significant']
record("False-positive rate when users are randomized but rows are sessions", "5%",
       f"{unit_level / sims_unit:.1%} (one row per user)", f"sessions treated as independent: {row_level / sims_unit:.1%}")

# 8. Ratio metrics: revenue per session, users randomized
rng = np.random.default_rng(1)
sims_ratio = 600
delta_hits = session_hits = 0


def ratio_arm(n):
    sessions = rng.integers(1, 30, n)
    per_session = [rng.exponential(5 * (1 + s / 30), s) for s in sessions]
    return np.array([x.sum() for x in per_session]), sessions.astype(float), np.concatenate(per_session)


for _ in range(sims_ratio):
    rev_c, ses_c, rows_c = ratio_arm(400)
    rev_t, ses_t, rows_t = ratio_arm(400)
    delta_hits += engine.ratio_metric_test(rev_c, ses_c, rev_t, ses_t)['significant']
    session_hits += engine.t_test(rows_c, rows_t)['significant']
record("False-positive rate for revenue per session (ratio metric)", "5%",
       f"{delta_hits / sims_ratio:.1%} (delta method)", f"t-test on sessions: {session_hits / sims_ratio:.1%}")

table = pd.DataFrame(rows)
print("| " + " | ".join(table.columns) + " |")
print("|" + "|".join("---" for _ in table.columns) + "|")
for row in table.itertuples(index=False):
    print("| " + " | ".join(map(str, row)) + " |")
