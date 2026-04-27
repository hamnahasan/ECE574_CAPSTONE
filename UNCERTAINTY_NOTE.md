# MC Dropout Uncertainty — Quick Reference

> A short note on how we actually computed the reliability diagram and ECE numbers, so the team can answer questions about the uncertainty experiment without ambiguity.

---

## Did we apply MC Dropout, or use the model "as is"?

**We applied MC Dropout.** The reliability diagram and the ECE values both come from MC-averaged predictions — not a single deterministic forward pass.

---

## The protocol, step by step

For every test (and Bolivia) chip:

1. Set the model to `eval()` mode — this freezes batch-norm / group-norm statistics.
2. **Re-enable dropout layers only** (custom `enable_dropout()` helper). Without this, dropout would be off at inference and we'd have a deterministic model.
3. Run the model **N = 20 times** on the same chip. Each pass samples a different sub-network (different neurons dropped), so each pass gives a slightly different prediction.
4. For each pixel:
   - **Mean water probability** = average of the 20 softmax outputs → this is the "best" probability estimate
   - **Predictive uncertainty** = variance across the 20 outputs → high variance = sub-networks disagree = uncertain region
5. Concatenate all chips' mean probabilities + ground-truth labels.
6. Sort pixels into **15 bins by predicted confidence** (0.00–0.067, 0.067–0.133, …, 0.933–1.00).
7. For each bin compute `accuracy_b - confidence_b`. The **count-weighted average of those gaps is ECE**.

---

## What we actually report

| Quantity | What it is | Test split | Bolivia split |
|---|---|---|---|
| **Mean prediction** | Average softmax over 20 MC passes | used for IoU/Dice + reliability | used for IoU/Dice + reliability |
| **Predictive uncertainty** | Variance over 20 MC passes | per-pixel maps in `results/figures/uncertainty_trimodal_test/` | per-pixel maps in `results/figures/uncertainty_trimodal_bolivia/` |
| **ECE** | Calibration error from the reliability diagram | **0.0273** | **0.0451** |

Both ECE values are below the literature "well-calibrated" threshold of 0.05, with test in the "excellent" range (<0.03). The slight loosening on Bolivia is expected — the model is more uncertain on out-of-distribution geography — but it's still trustworthy.

---

## Why MC matters here (vs reading off a single softmax)

If we **didn't** use MC Dropout we could still produce a reliability diagram from the model's normal softmax outputs — but it would only measure how calibrated the **point estimate** is. MC Dropout adds two things:

1. **A more principled probability estimate.** The mean of N stochastic forward passes is a Monte-Carlo approximation to the Bayesian posterior predictive (Gal & Ghahramani 2016). In practice, MC-averaged softmax is better calibrated than the single-shot softmax.
2. **A separate uncertainty signal.** The variance across passes is **epistemic uncertainty** — how much the model would change its mind given more training. This is what produces the per-pixel uncertainty maps. A single deterministic pass cannot give you this; you'd only have the point prediction.

For disaster response we need both: a reliable probability **and** a confidence on that probability.

---

## Where the code lives

- `src/utils/uncertainty.py` — `mc_predict()`, `compute_ece()`, `reliability_diagram()`, `evaluate_uncertainty()`
- `scripts/mc_uncertainty.py` — CLI that loads a checkpoint, builds the dataset, runs MC eval, saves JSON + figures
- `slurm/mc_uncertainty.sbatch` — runs the test-split version on Isaac
- `slurm/mc_uncertainty_bolivia.sbatch` — runs the Bolivia-split version on Isaac

---

## Two-sentence version for Q&A

> "Yes — we apply Monte Carlo Dropout: 20 stochastic forward passes per chip with dropout left active, then average the predictions and use that average for the reliability diagram. The variance across the 20 passes is what produces the per-pixel uncertainty maps; the mean is what gives us ECE 0.0273 on test and 0.0451 on Bolivia, both well-calibrated."
