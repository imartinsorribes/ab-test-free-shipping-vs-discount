"""
Generates a simulated dataset for an A/B test on e-commerce offers.

Scenario
--------
An online store wants to know which offer works best on the product page:

    A (control)   -> "10% off your order"
    B (treatment) -> "Free shipping on orders of 40 EUR or more"

Each visitor is randomly assigned to one of the two variants for 28 days.
The "true effect" is defined here, but the analysis treats it as unknown:
the goal is to check whether the statistical methods recover it correctly.

Usage:
    python src/generate_data.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
N_USERS = 90_000
N_DAYS = 28
START_DATE = "2026-09-01"

# Baseline purchase probability (variant A) by device
BASE_CR = {"mobile": 0.031, "desktop": 0.047}
# Multipliers on the purchase probability
USER_TYPE_MULT = {"new": 0.80, "returning": 1.35}
SOURCE_MULT = {"organic": 1.00, "paid": 0.85, "email": 1.40, "social": 0.75}
# True effect of variant B on conversion (heterogeneous by device)
LIFT_B = {"mobile": 1.17, "desktop": 1.04}

# Business economics
DISCOUNT_A = 0.10        # product discount in variant A
GROSS_MARGIN = 0.40      # gross margin on list price
SHIPPING_COST = 4.90     # shipping cost absorbed by the store in B
FREE_SHIPPING_MIN = 40.0  # minimum order for free shipping in B

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = ROOT / "data" / "experiment_data.csv"


def generate(n_users: int = N_USERS, seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    variant = np.where(rng.random(n_users) < 0.5, "B", "A")
    device = rng.choice(["mobile", "desktop"], size=n_users, p=[0.62, 0.38])
    user_type = rng.choice(["new", "returning"], size=n_users, p=[0.65, 0.35])
    source = rng.choice(
        ["organic", "paid", "email", "social"], size=n_users, p=[0.40, 0.30, 0.12, 0.18]
    )
    day = rng.integers(0, N_DAYS, size=n_users)
    date = pd.Timestamp(START_DATE) + pd.to_timedelta(day, unit="D")

    # Conversion probability of each visitor
    p = (
        pd.Series(device).map(BASE_CR).to_numpy()
        * pd.Series(user_type).map(USER_TYPE_MULT).to_numpy()
        * pd.Series(source).map(SOURCE_MULT).to_numpy()
    )
    p = np.where(variant == "B", p * pd.Series(device).map(LIFT_B).to_numpy(), p)
    converted = rng.random(n_users) < p

    # Basket value at list price (log-normal, mean ~52 EUR)
    basket = rng.lognormal(mean=np.log(52) - 0.125, sigma=0.5, size=n_users)
    # In B, some customers just below the threshold add items to
    # qualify for free shipping
    near_threshold = (variant == "B") & (basket >= 28) & (basket < FREE_SHIPPING_MIN)
    tops_up = near_threshold & (rng.random(n_users) < 0.45)
    basket = np.where(tops_up, rng.uniform(FREE_SHIPPING_MIN, 46, size=n_users), basket)
    basket = np.where(converted, basket, 0.0)

    # Revenue and profit per visitor
    revenue = np.where(variant == "A", basket * (1 - DISCOUNT_A), basket)
    cogs = basket * (1 - GROSS_MARGIN)
    shipping_paid = np.where(
        (variant == "B") & converted & (basket >= FREE_SHIPPING_MIN), SHIPPING_COST, 0.0
    )
    profit = revenue - cogs - shipping_paid

    return pd.DataFrame(
        {
            "user_id": np.arange(1, n_users + 1),
            "date": date.strftime("%Y-%m-%d"),
            "variant": variant,
            "device": device,
            "user_type": user_type,
            "source": source,
            "converted": converted.astype(int),
            "revenue": revenue.round(2),
            "profit": profit.round(2),
        }
    )


if __name__ == "__main__":
    df = generate()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Dataset saved to {OUTPUT_PATH} ({len(df):,} rows)")
