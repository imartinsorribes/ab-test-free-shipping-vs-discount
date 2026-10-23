# Methodology notes

Why each method in this project was chosen, and what the alternatives are.

## 1. Sample size, MDE and power

The test duration is set **before** launch from three inputs:

- **MDE (minimum detectable effect)**: the smallest lift worth acting on. Here, 10% relative on a 3.6% baseline.
- **α = 0.05**: probability of a false positive (declaring a winner when there is no effect).
- **Power = 80%**: probability of detecting the effect if it is really as large as the MDE.

For two proportions: `n per variant = (z_{1-α/2} + z_{power})² · [p₁(1−p₁) + p₂(1−p₂)] / (p₂ − p₁)²` → ~44,000 visitors per variant.
Halving the MDE roughly **quadruples** the sample size, because n scales with 1/MDE².

## 2. Sample Ratio Mismatch (SRM)

A chi-square goodness-of-fit test on the number of users per variant. If the split is 50/50 by design but the observed counts
are very unlikely under 50/50 (p < 0.001, a strict threshold because the test is run on every experiment), something is broken:
assignment bugs, bot filtering applied to one arm, redirects that lose users… With SRM, **no metric can be trusted**, even if it looks good.

## 3. Which test for which metric

| Metric | Type | Test | Why |
|---|---|---|---|
| Conversion | Proportion per visitor | Two-proportion z-test | Bernoulli outcome; large n makes the normal approximation valid |
| Revenue / profit per visitor | Continuous, ~96% zeros, long tail | Welch's t-test **+ bootstrap** | Welch does not assume equal variances; the bootstrap does not assume normality, a good check for skewed data |
| Average order value | Mean over buyers only | Welch's t-test | See caveat below |

**AOV caveat.** AOV is computed only over buyers, and B changes *who* buys. That subset is selected after randomisation,
so the comparison is not fully causal. This is why the decision relies on **profit per visitor**, which is computed over
all randomised users. (For ratio metrics computed per visitor, such as revenue per session, the **delta method** gives the correct variance.)

## 4. Multiple comparisons: Holm vs Bonferroni

Testing 8 segments at α = 0.05 each gives a ~34% chance of at least one false positive (1 − 0.95⁸).

- **Bonferroni**: multiply each p-value by m (number of tests). Controls the family-wise error rate (FWER), but is very conservative.
- **Holm**: sort p-values from smallest to largest and multiply the k-th by (m − k + 1), keeping the result monotone.
  It controls **the same FWER as Bonferroni under the same assumptions, but is always at least as powerful**:
  it never rejects fewer hypotheses. There is no reason to prefer plain Bonferroni over Holm.
- **Benjamini-Hochberg**: controls the *false discovery rate* (expected share of false positives among discoveries) instead of FWER.
  It is more permissive and suits large-scale screening (e.g. 100 metrics) where a few false leads are acceptable.

Even with correction, segment results are **exploratory**: in this project, "returning users" and "paid traffic" come out
significant although the simulated effect does not vary along those dimensions.

## 5. Bayesian analysis and expected loss

With a Beta(1, 1) prior and binomial data, the posterior of each conversion rate is `Beta(1 + conversions, 1 + non-conversions)`.
Sampling from both posteriors gives:

- **P(B > A)**: the probability that B is actually better. This is what stakeholders usually think a p-value means (it is not).
- **Expected loss** of choosing B: `E[max(p_A − p_B, 0)]`, the conversion we expect to lose by picking B *if B is in fact worse*,
  averaged over the posterior. It combines how likely a mistake is with how costly it would be.
  A common decision rule is to ship when the expected loss of the chosen variant falls below a "threshold of caring" (e.g. 0.01 pp).
  Here, the expected loss of choosing B is ~0 and that of choosing A is 0.48 pp.

## 6. When to stop a test

- **Fixed horizon (used here)**: compute the sample size, run until it is reached, analyse once. Simple and valid.
- **Why not stop when p < 0.05?** Checking every day and stopping at the first significant result (*peeking*) inflates the
  false-positive rate well above 5% (above 20% with daily checks over a month), because the p-value fluctuates and will cross 0.05 by chance.
- **If early stopping is needed**, use methods designed for it: **group sequential designs** with alpha spending
  (e.g. O'Brien-Fleming boundaries, which are very strict early on and close to 0.05 at the end), or **always-valid inference**
  (e.g. mSPRT, the approach behind Optimizely's Stats Engine), which allows continuous monitoring.
- **Practical rules**: run at least one or two full weeks to cover weekly seasonality, and watch out for **novelty effects**
  (users react to the change itself and the effect fades over time). This is why the cumulative-conversion plot is included.

## 7. CUPED (not used here)

**CUPED** (Controlled-experiment Using Pre-Experiment Data, Microsoft, 2013) reduces metric variance using a covariate measured
**before** the experiment, such as each user's spend in the previous 30 days:

```
θ = cov(Y, X) / var(X)
Y_cuped = Y − θ · (X − mean(X))
```

Because X is measured before randomisation, it is independent of the treatment, so the estimate stays unbiased while variance
drops by a factor of `1 − ρ²` (ρ = correlation between X and Y). With ρ = 0.5 the variance falls by 25%, which means the same power
with ~25% fewer users, or a shorter test.

**Why it is not used here:** the dataset has no pre-experiment data per user. Many visitors to a store are new and have no history,
so the gain would come mainly from returning users. With real data, it would be the first improvement to add.

## 8. Guardrail metrics

Metrics that are not optimised but that block the launch if they degrade: for this offer, **shipping cost per order** (checked via profit
per visitor), **return rate** and **cancelled orders** (monitored for 30 days after launch, since they happen after the purchase), and
**checkout latency/errors**. Guardrails catch wins on the primary metric that are paid for elsewhere.

## 9. Limitations

- **Simulated data**: this lets us validate the method against a known truth, but it does not capture real-world issues
  (bots, users on several devices, tracking gaps).
- **Visitor-level independence** is assumed; a user who visits on two devices could be counted in both arms.
- **Short-term metrics only**: long-term effects (repeat purchases, learning that free shipping exists) are not measured.
