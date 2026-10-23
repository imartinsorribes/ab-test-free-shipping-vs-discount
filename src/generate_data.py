"""
Genera un dataset simulado de un experimento A/B sobre ofertas en un e-commerce.

Escenario
---------
Una tienda online quiere saber qué oferta funciona mejor en la página de producto:

    A (control)     -> "10% de descuento en tu pedido"
    B (tratamiento) -> "Envío gratis en pedidos de 40 EUR o más"

Cada visitante se asigna aleatoriamente a una de las dos variantes durante 28 días.
El "efecto real" se define aquí, pero en el análisis se trata como desconocido:
el objetivo es comprobar si los métodos estadísticos lo detectan correctamente.

Uso:
    python src/generate_data.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
N_USERS = 90_000
N_DAYS = 28
START_DATE = "2026-09-01"

# Probabilidad de compra base (variante A) por dispositivo
BASE_CR = {"mobile": 0.031, "desktop": 0.047}
# Multiplicadores sobre la probabilidad de compra
USER_TYPE_MULT = {"nuevo": 0.80, "recurrente": 1.35}
SOURCE_MULT = {"organico": 1.00, "paid": 0.85, "email": 1.40, "social": 0.75}
# Efecto real de la variante B sobre la conversión (heterogéneo por dispositivo)
LIFT_B = {"mobile": 1.17, "desktop": 1.04}

# Economía del negocio
DISCOUNT_A = 0.10        # descuento sobre productos en la variante A
GROSS_MARGIN = 0.40      # margen bruto sobre el precio de lista
SHIPPING_COST = 4.90     # coste de envío que asume la tienda en B
FREE_SHIPPING_MIN = 40.0  # pedido mínimo para envío gratis en B

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = ROOT / "data" / "experiment_data.csv"


def generate(n_users: int = N_USERS, seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    variant = np.where(rng.random(n_users) < 0.5, "B", "A")
    device = rng.choice(["mobile", "desktop"], size=n_users, p=[0.62, 0.38])
    user_type = rng.choice(["nuevo", "recurrente"], size=n_users, p=[0.65, 0.35])
    source = rng.choice(
        ["organico", "paid", "email", "social"], size=n_users, p=[0.40, 0.30, 0.12, 0.18]
    )
    day = rng.integers(0, N_DAYS, size=n_users)
    date = pd.Timestamp(START_DATE) + pd.to_timedelta(day, unit="D")

    # Probabilidad de conversión de cada visitante
    p = (
        pd.Series(device).map(BASE_CR).to_numpy()
        * pd.Series(user_type).map(USER_TYPE_MULT).to_numpy()
        * pd.Series(source).map(SOURCE_MULT).to_numpy()
    )
    p = np.where(variant == "B", p * pd.Series(device).map(LIFT_B).to_numpy(), p)
    converted = rng.random(n_users) < p

    # Valor de la cesta a precio de lista (log-normal, media ~52 EUR)
    basket = rng.lognormal(mean=np.log(52) - 0.125, sigma=0.5, size=n_users)
    # En B, parte de los clientes que se quedan cerca del mínimo añaden
    # productos para llegar al envío gratis
    near_threshold = (variant == "B") & (basket >= 28) & (basket < FREE_SHIPPING_MIN)
    tops_up = near_threshold & (rng.random(n_users) < 0.45)
    basket = np.where(tops_up, rng.uniform(FREE_SHIPPING_MIN, 46, size=n_users), basket)
    basket = np.where(converted, basket, 0.0)

    # Ingresos y beneficio de cada visitante
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
    print(f"Dataset guardado en {OUTPUT_PATH} ({len(df):,} filas)")
