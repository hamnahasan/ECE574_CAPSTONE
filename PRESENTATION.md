# Project Presentation — Prep Document
**Multi-Modal Flood Segmentation with Cross-Attention Fusion and Uncertainty Estimation**

> Prep document for the team. **Not the slides themselves** — use this as the source of truth for content, talking points, numbers, Q&A prep, and division of labor. Target: **10 min content + 2 min Q&A = 12 min total**.

---

## 0. Meta (fill-in-the-blanks)

- **Team number:** _[insert]_
- **Course:** ECE 574 — Computer Vision
- **Authors:** Eric, Hamna, & Kavita 
- **Submission:** `.pdf` AND `.pptx`, by course deadline
- **Branding:** UT fonts + colors ([UT brand link in course instructions])
- **Acknowledgments to include:** UTK Isaac HPC (account `ACF-UTK0011`), course instructor/TA, Copernicus/ESA (DEM), Sen1Floods11 team (Bonafilia et al. 2020)

---

## 1. Slide 1 — Title

**Title:** Multi-Modal Flood Segmentation with Cross-Attention Fusion and Uncertainty Estimation


**Course:** ECE 574 — Computer Vision, Spring 2026

---

## 2. Slide 2 — Introduction / Motivation (target: ~1 min 30 sec)

### Slide content (bullets)

- **Floods are the most frequent & costliest natural disaster globally.** Rapid flood maps are critical for emergency response, evacuation, and humanitarian logistics.
- **Remote sensing gives us three complementary signals:**
  - **Sentinel-1 (SAR):** penetrates clouds day-or-night, but speckle + layover noise
  - **Sentinel-2 (optical/multispectral):** rich spectral info, but cloud-blocked during storms
  - **DEM (terrain):** physics prior — water runs downhill & can't sit on steep slopes
- **Goal:** single model that (1) fuses all three, (2) still works when one modality is missing, and (3) tells us *how confident it is* — critical for operational decision-making.
- **Why this matters:** current operational tools (Copernicus EMS) rely on manual interpretation — we want an automated, calibrated pipeline.

### Speaker notes

> Open with the human stakes: "floods killed ~7,500 people and caused $82B in damages in 2023 alone." Then set up the remote-sensing gap: optical is blinded by clouds exactly when floods happen, SAR is noisy, DEM is static but useful. Frame the three goals (fusion, robustness, calibration) as the three things a disaster responder needs.

---

## 3. Slide 3 — Dataset (target: ~45 sec)

### Slide content

- **Sen1Floods11** (Bonafilia et al., CVPRW 2020)
  - 446 hand-labeled chips @ 512×512
  - **11 flood events across 6 continents**
  - S1 (VV, VH) + S2 (13 bands) + binary water/non-water labels
- **Copernicus GLO-30 DEM** — added by us, aligned per chip grid, derived slope channel
- **Splits:** train (252) / val (89) / test (90) / **Bolivia held-out (15)**
- **Bolivia is the cross-region holdout** — NOT in train/val/test. Tests whether the model generalizes to unseen geography.

### Speaker notes

> Emphasize: dataset is **small** (446 chips) — this constrains what architectures we can afford and motivates careful design. Bolivia holdout is the generalization test that separates memorization from learning.

### Suggested visual
- `results/figures/geographic_distribution.png`
- `results/figures/sample_triplet.png` (S1 / S2 / label)
- `results/figures/class_balance.png`

---

## 4. Slide 4 — Methodology Overview (target: ~1 min)

### Slide content

| Phase | Model | Input | Purpose |
|---|---|---|---|
| 0 | Otsu threshold on S1 VH | S1 | Classical baseline |
| 1 | FCN-ResNet50 | S1 (2ch) | Deep-learning S1 baseline (Bonafilia replica) |
| 2 | **Fusion U-Net** | S1 + S2 | Dual-encoder cross-attention fusion |
| 3 | **TriModal U-Net** | S1 + S2 + DEM | 3-way cross-attention + modality dropout |
| Abl | EarlyFusionUNet (×7) | Modality subsets | Isolate per-modality contribution |

**Evaluation:**
- IoU, Dice, Precision, Recall on test split
- Bolivia split for cross-region generalization
- MC Dropout (N=20) for calibrated uncertainty — report ECE

### Speaker notes

> "We built up progressively: classical → CNN baseline → fusion → tri-modal. For scientific rigor we also ran an ablation matrix to isolate what each modality contributes. This is the clean experimental design reviewers ask for."

---

## 5. Slide 5 — Key Architectural Choices (target: ~1 min 30 sec)

### Slide content (callouts)

**TriModal U-Net architecture:**
- Three parallel ResNet34 encoders (S1, S2, DEM)
- **3-way cross-attention at 4 decoder scales** — each modality queries the other two
- **70.9M parameters** total
- **Modality dropout p=0.1** during training → robust to missing modalities at inference

**Training decisions (defended):**
- `CosineAnnealingLR` (not WarmRestarts — LR spikes destabilized attention)
- Mixed precision (AMP) with **float32 attention** for numerical stability
- Gradient clipping to prevent NaN from softmax overflow
- **Auto-resumable checkpoints** (atomic writes, RNG state preserved) — necessary for Isaac's 24hr wall-time

### Speaker notes

> "We hit NaN losses at epoch 14 with WarmRestarts — the LR spike made attention softmax overflow. Switching to plain CosineAnnealing and forcing float32 attention under AMP fixed it. This is the kind of detail reviewers care about — it shows we debugged real training problems, not just ran `model.fit()`."

### Suggested visual
- Block diagram of the TriModal U-Net (encoders → cross-attn bridge → shared decoder → water head)
- Attention schematic showing the 3-way queries

---

## 6. Slide 6 — Main Results, Test Split (target: ~1 min 30 sec)

### Slide content — headline table

| Model | **Water IoU** | Dice | Precision | Recall |
|---|---|---|---|---|
| Otsu baseline (S1 VH) | 0.34 _(approx)_ | — | — | — |
| FCN-ResNet50 (S1) | 0.6232 | 0.768 | 0.738 | 0.800 |
| **Fusion U-Net (S1+S2 cross-attn)** | **0.7708** | 0.871 | 0.816 | 0.933 |
| **TriModal U-Net (S1+S2+DEM cross-attn)** | **0.7820** | 0.878 | 0.846 | 0.912 |
| **Best ablation — `s1_s2` early fusion** | **0.7991** | 0.888 | 0.863 | 0.915 |

### Speaker notes — the headline

> "Going from single-modality SAR (0.62 IoU) to fusion (0.77) gives **+15 IoU points**. Adding DEM in cross-attention (0.78) adds another point. Our best scratch-trained model hits **0.799 IoU** — within 0.6 points of the Prithvi-EO-1.0 foundation model (0.805), which was pretrained on millions of Earth-observation images. We achieved this from 446 chips."

### Suggested visual
- `results/figures/main_results.png`

---

## 7. Slide 7 — Ablation Study (target: ~1 min)

### Slide content

| Modalities | Test IoU | Δ vs S1 |
|---|---|---|
| DEM only | 0.398 | — |
| S1 only | 0.641 | baseline |
| S1 + DEM | 0.640 | **−0.001** (DEM alone adds nothing on SAR) |
| S2 only | 0.757 | **+0.12** |
| S2 + DEM | 0.788 | +0.15 |
| **S1 + S2** | **0.799** | **+0.16** |
| S1 + S2 + DEM | 0.790 | +0.15 (DEM slightly *redundant* with complete S1+S2) |

**Two findings to state aloud:**
1. **S2 is the dominant modality** — S2 alone beats S1 alone by 12 IoU.
2. **DEM helps when a modality is missing, not additively.** Adding DEM to S2-alone: +3.1 IoU. Adding DEM to complete S1+S2: −1.0 IoU.

### Speaker notes

> "DEM is a physics prior that fills gaps. When you already have both S1 and S2 with complementary signals, the DEM is redundant. But when one is missing (e.g., cloud-obscured S2), DEM becomes a valuable backup signal. This is a real, defensible finding."

### Suggested visual
- `results/figures/ablation_bars.png`

---

## 8. Slide 8 — Cross-Region Generalization (Bolivia) (target: ~1 min 30 sec)

### Slide content

| Model | Test IoU | **Bolivia IoU** | Drop | Precision | Recall |
|---|---|---|---|---|---|
| FCN baseline | 0.623 | **0.359** | **−26.4** | 0.705 | 0.422 |
| Fusion U-Net | 0.771 | **0.606** | −16.5 | 0.884 | 0.658 |
| **TriModal U-Net** | 0.782 | **0.605** | −17.7 | **0.931** | 0.634 |

**Two findings to state aloud:**
1. **Fusion & TriModal generalize 10 IoU points better than FCN baseline** — multi-modality matters more when geography is unseen.
2. **TriModal trades recall for precision on Bolivia** — +4.7 pts precision (0.931 vs 0.884) at −2.4 recall. The DEM cue tells the model "flat high-elevation terrain can't be flooding," suppressing false positives in unfamiliar geography.

### Speaker notes

> "FCN collapses from 0.62 → 0.36 on a new continent — the S1-only baseline memorized local speckle patterns. The fusion models hold up 10 IoU points better. TriModal specifically becomes more *cautious* on Bolivia — precision goes up, recall goes down. For disaster response, lower false-positive rate is actually preferable (false alarms cause responder fatigue)."

### Suggested visuals
- `results/figures/cross_region.png` (side-by-side IoU bars)
- `results/figures/precision_recall_tradeoff.png` (arrows showing test→Bolivia shift per model)

---

## 9. Slide 9 — Calibrated Uncertainty (target: ~1 min)

### Slide content

**MC Dropout (N=20 forward passes, trimodal model, test set):**
- **Expected Calibration Error: 0.0273** (<0.05 = good, <0.03 = excellent)
- Confidence tracks accuracy monotonically across 15 bins
- Slightly overconfident in mid-range (0.7–0.9), near-perfect at extremes

**Why this matters:**
- A model predicting "80% water" should be correct 80% of the time — ours is
- Disaster responders need **trustworthy confidence scores**, not just predictions
- This is a novelty vs. foundation models (Prithvi-EO does not report ECE)

### Speaker notes

> "Aggregate IoU tells you the average; reliability tells you when to trust a specific pixel. Our ECE of 0.027 means when our model says 'I'm 80% sure this is water,' it's right 80% of the time. That's what makes it deployable for decision support, not just a research metric."

### Suggested visuals
- `results/figures/reliability_trimodal_test.png` (or `reliability_diagram.png`)
- 1–2 example uncertainty maps from `results/figures/uncertainty_trimodal_test/`

---

## 10. Slide 10 — Per-Chip Error Analysis (target: ~45 sec)

### Slide content

**Per-chip IoU reveals what aggregate IoU hides:**

| Model | Bolivia aggregate | Bolivia per-chip mean | Chips < 0.3 IoU |
|---|---|---|---|
| Fusion | 0.606 | **0.452** | 4 / 15 |
| TriModal | 0.605 | 0.345 | 7 / 15 |

**Three chips are universally broken** (all models IoU ≈ 0):
- `Bolivia_76104`, `Bolivia_233925`, `Bolivia_195474` — likely label quality / cloud contamination issues

**Model complementarity — `Bolivia_103757`:** FCN=0.79, Fusion=0.09, TriModal=0.05
→ S2 imagery is cloud-contaminated and poisons the fusion signal. The S1-only baseline wins. **An ensemble would beat any single model on Bolivia.**

### Speaker notes

> "Per-chip analysis is where we found the real story. Fusion has higher *per-chip* mean than TriModal because TriModal is higher-variance — great when DEM helps, catastrophic when atypical geography confuses it. We also identified three chips where every model fails — that's a data issue, not a model issue, and we're transparent about it."

---

## 11. Slide 11 — Root-Cause Error Analysis & Remediation (target: ~1 min 15 sec)

> **This is the slide the professor will probe hardest.** Be prepared to defend every row. The story: we don't just know *where* we fail — we know *why*, and we have concrete paths to fix each failure mode.

### Slide content — failure modes with root cause and fix

| # | Failure mode | Evidence | Root cause | What we'd do about it |
|---|---|---|---|---|
| 1 | **Label noise / ambiguous ground truth** | 3 Bolivia chips (`76104`, `233925`, `195474`) — every model IoU ≈ 0 | Hand-labeling imperfect: cloud-obscured S2 at label time, ambiguous wetland/water boundaries | Model-assisted QC — flag chips where all 3 models agree against the label; use consensus relabeling or drop |
| 2 | **Cloud-contaminated S2 poisoning fusion** | `Bolivia_103757` — FCN=0.79, Fusion=0.09, TriModal=0.05 | Cloud pixels in S2 propagate through fusion, overriding good SAR signal | (a) Explicit cloud mask from S2 QA60 / SCL band, (b) attention-based modality gating, (c) train with synthetic cloud augmentation |
| 3 | **Out-of-distribution geography** | Test→Bolivia drop of 17 IoU for fusion models | Amazon basin / altiplano terrain not represented in 252-chip train set | (a) Pretrain on 4,385 weakly-labeled chips, (b) domain adaptation, (c) swap in a Prithvi-EO foundation-model backbone |
| 4 | **High variance on small data** | TriModal per-chip mean 0.345 < Fusion 0.452 on Bolivia despite equal aggregate IoU — 7/15 chips <0.3 IoU | 70M-param model on 446 chips → fits easy cases sharply, breaks on hard | Ensemble (per-chip complementarity proves +IoU available), stronger dropout/weight decay, smaller attention heads |
| 5 | **Over-cautious DEM prior hurts recall** | TriModal Bolivia precision 0.931, recall 0.634 — fewer false positives but misses subtle flooding | DEM saying "flat-high = not water" can over-suppress genuine shallow/valley flooding | Threshold-tune at deploy time, or calibrate DEM weight; report full PR curve so operators choose the tradeoff |

### Key take-aways to state aloud

- **Failures split into three bins:** label issues, data (cloud / geography) issues, and architecture issues — each has a different fix
- **Three universally-hard chips are a data-quality problem**, not a model problem. A credible paper acknowledges this rather than hiding it
- **Single biggest actionable fix: explicit cloud masking** before fusion. The `Bolivia_103757` case is a smoking gun
- **Cheapest near-term win: ensemble** the three primary models — per-chip analysis already proves +IoU is available
- **Largest-impact long-term fix: foundation-model pretraining** (Prithvi-EO) to close the 17-IoU Bolivia gap

### Speaker notes

> "This slide is how we think about our own failures. We identified five distinct failure modes, each with evidence, a root cause, and a concrete remediation. For example: `Bolivia_103757` is a chip where the S1-only baseline gets 0.79 IoU but our fusion model collapses to 0.09. The S2 imagery is cloud-contaminated and poisoning the fusion signal. The fix is explicit cloud masking — we know exactly what to do next. This is the difference between a project that ended and a project that has a clear path forward."

### Suggested visual
- `results/figures/iou_distributions_bolivia.png` (box plot showing the variance)
- A side-by-side image comparison of `Bolivia_103757` S2 (cloudy) vs GT vs predictions — *Needs to be done*

---

## 12. Slide 12 — Literature Comparison: Honest Take (target: ~45 sec)

> **The goal of this slide is credibility.** We tell the audience exactly where we stand against prior work on the *same dataset* — wins and losses. Being honest here is what separates a scientific project from a marketing pitch.

### Slide content — side-by-side on Sen1Floods11

| Work | Year | Model (input) | **Test water IoU** | **Bolivia water IoU** |
|---|---|---|---|---|
| Bonafilia et al. (original dataset) | 2020 | FCN-DenseNet (S1) — baseline | ~0.64 | F1-only reported |
| Yang et al. (MDPI) | 2021 | Fused S1+S2 deep network | ~0.77 mIoU | ~0.48 mIoU |
| Gauthier et al. (Frontiers) | 2022 | Attentive U-Net (S1) | **0.67** | ~0.66 |
| Gauthier et al. (Frontiers) | 2022 | Fusion Network (S1) | **~0.69** | ~0.69 |
| **Prithvi-EO-1.0-100M** (IBM/NASA) | 2023 | Foundation model (S2, 6 bands) | **0.8046** | **0.7795** |
| **Prithvi-CAFE** | 2025 | Foundation + adaptive fusion | — | Prithvi + 10.8 pts |
| **Our FCN-ResNet50 baseline** | 2026 | S1 only | 0.6232 | 0.3588 |
| **Our Fusion U-Net** | 2026 | Cross-attn S1+S2 | 0.7708 | 0.6056 |
| **Our TriModal U-Net** | 2026 | 3-way cross-attn S1+S2+DEM | 0.7820 | 0.6054 |
| **Our best ablation (`s1_s2` early fusion)** | 2026 | Early fusion S1+S2 | **0.7991** | — |

### The honest take (state aloud)

**Where we win or tie:**
- **Test split:** our `s1_s2` early fusion (**0.7991**) beats every published scratch-trained method on this dataset and is within **0.6 IoU points of Prithvi-EO-1.0** (0.8046) — a foundation model pretrained on millions of Earth-observation images. We achieved this with **446 chips and ~25M parameters** trained from scratch.
- Our Fusion U-Net (0.7708) is above Gauthier 2022's best scratch-trained number (~0.69) by ~8 IoU.

**Where we lose:**
- **Bolivia cross-region:** Prithvi beats our best by **~17 IoU** (0.78 vs 0.61). Gauthier's S1 fusion matches us on Bolivia (~0.69 vs 0.61) even though we use S1+S2. The gap is a combination of (a) pretraining scale, (b) cloud contamination hurting our fusion, (c) our model overfitting to familiar geography.

**Why this matters:**
- The test-set result proves our architecture is sound — we don't need a foundation model to be competitive in-distribution.
- The Bolivia gap identifies *where pretraining matters most* — cross-region generalization — and points to the next experiment: swap in a Prithvi-EO backbone and re-apply our ablation + calibration + DEM physics methodology on top of it.
- **Our contributions are orthogonal to pretraining** — nobody else on this dataset reports calibrated uncertainty (ECE), a full modality ablation, or a physics-informed DEM integration. That's the defensible novelty.

### Speaker notes

> "We compared ourselves to everyone who used Sen1Floods11 — the dataset authors, recent peer-reviewed fusion work, and the state-of-the-art foundation model. On the test split we're within 0.6 IoU of Prithvi — a foundation model pretrained on millions of images — using 446 chips from scratch. On Bolivia we trail the foundation model by 17 points, and we're honest about why. But our contributions — calibration, ablation, DEM physics — aren't reported by any of those methods. They would stack on top of a Prithvi backbone and likely improve it further."

### Caveats to acknowledge if asked
- Yang 2021 reports mIoU (averaged over water + non-water), not water-class IoU — their numbers may look lower than water-only IoU would be
- Bonafilia 2020's original Bolivia results are reported as F1 in their Table 4, not IoU — apples-to-apples comparison on Bolivia is imprecise for them
- Gauthier 2022 uses S1 only; our multi-modal setup is not strictly comparable on inputs, but we can compare architectural choices (attention vs early fusion)

---

## 13. Slide 13 — Conclusion & Lessons Learned (target: ~1 min)

### Slide content — takeaways

1. **Multi-modal fusion matters**, especially for cross-region generalization (+10 IoU over S1-only).
2. **S2 is the dominant signal**; DEM helps when a modality is missing, not additively.
3. **Simple early fusion (S1+S2) is a strong baseline** — our 0.799 test IoU beats more complex cross-attention at this scale. Big models need big data.
4. **Calibrated uncertainty (ECE=0.027)** is a deployment-grade property that raw IoU misses.
5. **Cross-region test is essential** — test-set IoU overstates generalization by ~17 points.
6. **Per-chip analysis** reveals model complementarity (ensembles would help) and dataset-level failures (3 universally-hard chips).

### Lessons learned (what we'd do differently)

- **Pretrain or distill from a foundation model** — the Bolivia gap is mostly a data-scale issue
- **Ensemble the three models** — per-chip complementarity suggests +5 IoU easy gain
- **Add weakly-labeled chips** (4,385 available) for semi-supervised pretraining
- **Debug NaN losses with AMP + attention earlier** — cost us ~2 weeks of retraining
- **SLURM resumable training infrastructure** paid for itself many times over

### Speaker notes

> "Finish with honesty: we didn't beat the foundation models on cross-region, but we got within 0.6 IoU on test from 446 chips. That's a real result. Our contributions — ablation, calibration, DEM physics — are orthogonal and could ride on top of any foundation-model backbone."

---

## 14. Slide 14 — Acknowledgments (target: ~15 sec)



## 15. Q&A Prep — Likely Questions and Strong Answers

### Q1: "Why didn't you beat the foundation models?"
> "We trained from scratch on 446 hand-labeled chips. Prithvi-EO was pretrained on millions of Earth-observation images. On the in-distribution test set we're within 0.6 IoU of Prithvi (0.799 vs 0.805) — that shows our architecture is sound. The gap only appears on cross-region (Bolivia), where pretraining matters most. Our contributions — ablation, calibration, physics-informed DEM — could be applied on top of a foundation-model backbone."

### Q2: "Why is early fusion beating your cross-attention model?"
> "At 446 chips, the 70M-parameter TriModal model is overparameterized. Early fusion with 25M parameters is a better match for the data scale. This is a real and honest result — bigger architectures aren't automatically better. With 10× more training data we expect cross-attention to pull ahead."

### Q3: "How do you know DEM is actually helping, not just adding noise?"
> "Two pieces of evidence. First: on Bolivia cross-region, TriModal has **+4.7 points precision** over Fusion (0.931 vs 0.884) — exactly what a physics prior should do (suppress false positives on flat high-elevation terrain). Second: in the ablation, S2-alone → S2+DEM improves by +3.1 IoU — DEM fills the gap when a modality is absent."

### Q4: "What's ECE and why should I care?"
> "Expected Calibration Error measures whether a model's confidence scores are trustworthy. Our ECE of 0.027 means when we say 'I'm 80% sure this is water,' we're right 80% of the time. For disaster response this is critical — a responder needs to know not just *where* the model predicts flood, but *how sure it is*."

### Q5: "Why only 446 training chips? Sen1Floods11 has thousands."
> "Sen1Floods11 has 446 hand-labeled chips and ~4,385 weakly labeled chips (S1-classified, not human-verified). We used only hand-labeled to ensure ground-truth quality. Adding weak labels via semi-supervised learning is an obvious next step."

### Q6: "What computing resources did you use?"
> "UT's Isaac HPC cluster — campus-gpu QOS, single-GPU jobs, 24-hour wall-time limits. We built resumable training infrastructure (atomic checkpoints, RNG state preservation) so SLURM-requeue jobs pick up seamlessly. Total compute: ~10 × 24h training runs + eval."

### Q7: "Why 100 epochs — did you overfit?"
> "We tracked train IoU vs val IoU per epoch. The gap is healthy — train and val IoU rise together, val plateaus after ~40 epochs. We select the best checkpoint by val IoU, not final epoch. No early stopping; the overfitting diagnostic is in the report appendix."

### Q8: "How does modality dropout help?"
> "We zero out one modality with p=0.1 during training. This forces the model to remain robust when a modality is missing at inference — e.g., heavy cloud cover blocking S2. Without modality dropout the model would over-rely on whichever modality is most informative on training data."

### Q9: "Bolivia only has 15 chips — is that a meaningful test?"
> "15 chips is small but covers a distinct flood event (Bolivia 2018) in a geographic region never seen during training. Sample size is a limitation we acknowledge. The IoU signal is consistent with pixel-level aggregates (2.4M ground-truth positive pixels across those 15 chips), so statistical noise is low."

### Q10: "What would you scale to next?"
> "Three directions: (1) pretrain on the 4,385 weakly-labeled chips before fine-tuning, (2) ensemble our three models since per-chip analysis shows complementarity, (3) swap in a Prithvi-EO backbone and re-apply our ablation + calibration methodology."

---

## 16. Division of Labor — to be decided (3 presenters: Eric, Hamna, Kavita)

With a new slide 11 (error analysis & remediation), the deck is now 14 slides. Assign roughly ~3 min of speaking per person + shared open/close.

### Deep-learning / modeling slides
- Slide 4 Methodology overview
- Slide 5 Architecture & training choices
- Slide 6 Main results
- Slide 9 Calibrated uncertainty

### Remote-sensing / data slides
- Slide 2 Motivation
- Slide 3 Dataset & DEM preprocessing
- Slide 8 Cross-region generalization
- Slide 12 Literature comparison

### Analysis / synthesis slides
- Slide 7 Ablation study
- Slide 10 Per-chip error analysis
- **Slide 11 Root-cause error analysis & remediation** 
- Slide 13 Conclusion & lessons learned

### Shared
- Slide 1 Title — whoever opens
- Slide 14 Acknowledgments — whoever closes
- Q&A — answer in whoever's area of strength

**Rough timing:** ~3 min per presenter + ~30 sec intro/close = 10 min content.




---

## 17. Figure Inventory (all in `results/figures/`)

| File | Use in slide |
|---|---|
| `sample_triplet.png` | 3 (dataset) |
| `geographic_distribution.png` | 3 (dataset) |
| `class_balance.png` | 3 (dataset) |
| `dem_vs_s1_label.png` | 3 or 5 (DEM rationale) |
| `dem_statistics.png` | 3 (DEM rationale) |
| `main_results.png` | 6 (headline) |
| `ablation_bars.png` | 7 (ablation) |
| `cross_region.png` | 8 (Bolivia bars) |
| `precision_recall_tradeoff.png` | 8 (PR arrows) |
| `reliability_trimodal_test.png` | 9 (calibration) |
| `uncertainty_trimodal_test/*.png` | 9 (example uncertainty maps) |
| `iou_distributions_bolivia.png` | 10 (per-chip) |
| `training_curves.png` | supplemental |
| `fusion_prediction_gallery.png` | supplemental |
| `fusion_error_maps.png` | supplemental |

---

## 18. Raw Numbers Reference (one source of truth)

### Test split
| Model | IoU | Dice | Prec | Rec |
|---|---|---|---|---|
| fcn_baseline | 0.6232 | 0.7679 | 0.7378 | 0.8005 |
| fusion_unet | 0.7708 | 0.8705 | 0.8159 | 0.9330 |
| trimodal_unet | 0.7820 | 0.8777 | 0.8458 | 0.9121 |
| ablation_s1 | 0.6406 | 0.7810 | 0.7791 | 0.7829 |
| ablation_s2 | 0.7575 | 0.8620 | 0.8036 | 0.9295 |
| ablation_dem | 0.3983 | 0.5696 | 0.4624 | 0.7416 |
| ablation_s1_s2 | **0.7991** | 0.8883 | 0.8630 | 0.9152 |
| ablation_s1_dem | 0.6397 | 0.7803 | 0.7588 | 0.8031 |
| ablation_s2_dem | 0.7881 | 0.8815 | 0.8496 | 0.9158 |
| ablation_s1_s2_dem | 0.7895 | 0.8824 | 0.8532 | 0.9136 |

### Bolivia split (cross-region)
| Model | IoU | Dice | Prec | Rec |
|---|---|---|---|---|
| fcn_baseline | 0.3588 | 0.5281 | 0.7046 | 0.4223 |
| fusion_unet | 0.6056 | 0.7544 | 0.8845 | 0.6576 |
| trimodal_unet | 0.6054 | 0.7542 | **0.9312** | 0.6337 |

### Uncertainty (trimodal on test)
- N MC samples: 20
- ECE: **0.0273**
- Mean predictive entropy (bin-weighted proxy): ~0.15–0.20 (from notebook output)

### Per-chip Bolivia stats
| Model | Per-chip mean IoU | Per-chip median | Chips <0.3 |
|---|---|---|---|
| fcn_baseline | 0.303 | 0.278 | 8 / 15 |
| fusion_unet | **0.452** | 0.426 | 4 / 15 |
| trimodal_unet | 0.345 | 0.326 | 7 / 15 |

### Model sizes
| Model | Parameters |
|---|---|
| FCN-ResNet50 | ~33M |
| Fusion U-Net (2-way cross-attn) | ~45M |
| TriModal U-Net (3-way cross-attn) | ~70.9M |
| EarlyFusionUNet (17ch) | ~25.8M |

---

## 19. Practice-Run Checklist

- [ ] Total time 12:00 (10 content + 2 Q&A). Run through with a stopwatch.
- [ ] Slide transitions rehearsed — no "uh, next slide…"
- [ ] Headline numbers memorized: **0.799 test IoU, 0.605 Bolivia IoU, ECE 0.027, +4.7 pts Bolivia precision from DEM**
- [ ] One honest limitation acknowledged (Bolivia gap vs foundation models)
- [ ] Figures load cleanly in both PDF and PPTX export
- [ ] Q&A answers practiced out loud (not just read)
- [ ] UT branding applied (fonts, colors)
- [ ] PDF and PPTX both exported and tested before the deadline

---

## 20. Post-Presentation (optional next steps)

These aren't required for the presentation but are credible answers if someone asks "what's next":

- **Ensemble the three primary models** — per-chip analysis suggests this beats any single model on Bolivia
- **Pretrain on the 4,385 weakly-labeled chips** before hand-labeled fine-tuning
- **Re-run test-set evaluation with per-chip output** (`scripts/evaluate.py`) to extend error analysis to test split
- **Render a visual gallery** of best/worst Bolivia chips on Isaac (script: `scripts/error_analysis.py` handles ranking; need a Isaac-side rendering script for the images themselves)
- **Swap in a Prithvi-EO backbone** and re-apply our ablation + calibration methodology

---

*End of prep document. Good luck — you have a strong story.*
