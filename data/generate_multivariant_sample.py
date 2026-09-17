"""
Generates data/multivariant_checkout_test.csv

Synthetic three-arm checkout experiment built to exercise the features the
other samples cannot: more than two variants, a pre-experiment covariate for
CUPED, and a guardrail metric that one variant damages.

    python data/generate_multivariant_sample.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

rng = np.random.default_rng(2024)
n = 6000

variant = rng.choice(['control', 'one_page_checkout', 'express_pay'], n)
pre_revenue = rng.gamma(shape=2.0, scale=40.0, size=n)            # spend in the 30 days before assignment

# True effects: one_page_checkout +4% revenue, no latency cost
#               express_pay       +6% revenue, but slower pages
revenue_lift = pd.Series(variant).map({'control': 1.00, 'one_page_checkout': 1.04, 'express_pay': 1.06}).to_numpy()
revenue = (0.8 * pre_revenue + rng.normal(20, 12, n)).clip(min=0) * revenue_lift

convert_rate = pd.Series(variant).map({'control': 0.080, 'one_page_checkout': 0.092, 'express_pay': 0.095}).to_numpy()
converted = (rng.random(n) < convert_rate).astype(int)

latency_shift = pd.Series(variant).map({'control': 0, 'one_page_checkout': 0, 'express_pay': 90}).to_numpy()
page_load_ms = rng.normal(820, 150, n).clip(min=200) + latency_shift

df = pd.DataFrame({
    'user_id': np.arange(1, n + 1),
    'variant': variant,
    'pre_revenue': pre_revenue.round(2),
    'revenue': revenue.round(2),
    'converted': converted,
    'page_load_ms': page_load_ms.round(0).astype(int),
})
out = Path(__file__).parent / 'multivariant_checkout_test.csv'
df.to_csv(out, index=False)
print(f"Wrote {len(df):,} rows to {out}")
