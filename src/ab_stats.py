"""
Funciones estadísticas para analizar un test A/B.

Incluye el enfoque frecuentista (tamaño muestral, SRM, z-test, Welch, bootstrap,
corrección por comparaciones múltiples) y el bayesiano (Beta-Binomial).
"""

from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass
class TestResult:
    metric: str
    value_a: float
    value_b: float
    abs_diff: float
    rel_lift: float
    ci_low: float
    ci_high: float
    p_value: float

    @property
    def significant(self) -> bool:
        return self.p_value < 0.05


def sample_size_proportions(
    p_base: float, mde_rel: float, alpha: float = 0.05, power: float = 0.80
) -> int:
    """Visitantes necesarios POR VARIANTE para detectar un lift relativo `mde_rel`."""
    p_new = p_base * (1 + mde_rel)
    z_alpha = stats.norm.ppf(1 - alpha / 2)
    z_beta = stats.norm.ppf(power)
    variance = p_base * (1 - p_base) + p_new * (1 - p_new)
    return int(np.ceil((z_alpha + z_beta) ** 2 * variance / (p_new - p_base) ** 2))


def srm_check(n_a: int, n_b: int, expected_share_a: float = 0.5) -> float:
    """Sample Ratio Mismatch: p-valor chi-cuadrado del reparto de tráfico.

    Un p-valor muy bajo (< 0.001) indica que la asignación está rota y que los
    resultados del test no son fiables.
    """
    total = n_a + n_b
    expected = [total * expected_share_a, total * (1 - expected_share_a)]
    return stats.chisquare([n_a, n_b], f_exp=expected).pvalue


def two_proportion_ztest(
    x_a: int, n_a: int, x_b: int, n_b: int, alpha: float = 0.05, metric: str = "conversion"
) -> TestResult:
    """Z-test de dos proporciones (H0: p_a == p_b) con IC de la diferencia."""
    p_a, p_b = x_a / n_a, x_b / n_b
    p_pool = (x_a + x_b) / (n_a + n_b)
    se_pool = np.sqrt(p_pool * (1 - p_pool) * (1 / n_a + 1 / n_b))
    z = (p_b - p_a) / se_pool
    p_value = 2 * stats.norm.sf(abs(z))

    se_diff = np.sqrt(p_a * (1 - p_a) / n_a + p_b * (1 - p_b) / n_b)
    z_crit = stats.norm.ppf(1 - alpha / 2)
    diff = p_b - p_a
    return TestResult(
        metric, p_a, p_b, diff, diff / p_a, diff - z_crit * se_diff, diff + z_crit * se_diff, p_value
    )


def welch_ttest(
    a: np.ndarray, b: np.ndarray, alpha: float = 0.05, metric: str = "mean"
) -> TestResult:
    """T-test de Welch para la diferencia de medias (no asume varianzas iguales)."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    res = stats.ttest_ind(b, a, equal_var=False)
    ci = res.confidence_interval(confidence_level=1 - alpha)
    diff = b.mean() - a.mean()
    return TestResult(metric, a.mean(), b.mean(), diff, diff / a.mean(), ci.low, ci.high, res.pvalue)


def bootstrap_mean_diff(
    a: np.ndarray, b: np.ndarray, n_boot: int = 5_000, alpha: float = 0.05, seed: int = 0
) -> tuple[float, float, np.ndarray]:
    """IC bootstrap (percentil) de mean(b) - mean(a).

    Útil para métricas muy asimétricas como el ingreso por visitante, donde la
    mayoría de valores son 0.
    """
    rng = np.random.default_rng(seed)
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    boot = np.empty(n_boot)
    for i in range(n_boot):
        boot[i] = rng.choice(b, b.size).mean() - rng.choice(a, a.size).mean()
    low, high = np.quantile(boot, [alpha / 2, 1 - alpha / 2])
    return low, high, boot


def bayesian_proportions(
    x_a: int, n_a: int, x_b: int, n_b: int, n_samples: int = 200_000, seed: int = 0
) -> dict:
    """Modelo Beta-Binomial con prior uniforme Beta(1, 1).

    Devuelve la probabilidad de que B sea mejor que A, el intervalo de
    credibilidad del lift relativo y la pérdida esperada de elegir cada variante.
    """
    rng = np.random.default_rng(seed)
    post_a = rng.beta(1 + x_a, 1 + n_a - x_a, n_samples)
    post_b = rng.beta(1 + x_b, 1 + n_b - x_b, n_samples)
    lift = post_b / post_a - 1
    return {
        "prob_b_better": float((post_b > post_a).mean()),
        "lift_mean": float(lift.mean()),
        "lift_ci": tuple(np.quantile(lift, [0.025, 0.975])),
        "expected_loss_a": float(np.maximum(post_b - post_a, 0).mean()),
        "expected_loss_b": float(np.maximum(post_a - post_b, 0).mean()),
        "samples_a": post_a,
        "samples_b": post_b,
    }


def holm_correction(p_values: list[float]) -> np.ndarray:
    """Ajuste de Holm-Bonferroni para comparaciones múltiples."""
    p = np.asarray(p_values, dtype=float)
    order = np.argsort(p)
    m = len(p)
    adjusted = np.empty(m)
    running_max = 0.0
    for rank, idx in enumerate(order):
        running_max = max(running_max, (m - rank) * p[idx])
        adjusted[idx] = min(running_max, 1.0)
    return adjusted
