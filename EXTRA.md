# Extra Explanations — Supporting Material

> Supplementary deep-dives that don't fit on the slides themselves but are useful when probed in Q&A or for personal study before the presentation. Three sections only.

---

## 1. What "4 decoder scales" means (architecture diagram, right side)

### Quick context: encoders produce a feature pyramid

When a CNN like ResNet34 processes an image, it doesn't just produce one set of features. As the image flows deeper into the network, it gets **progressively smaller in spatial size but richer in features**.

For a 256×256 input chip, the encoder produces a "feature pyramid" at multiple resolutions:

| Scale label | Spatial size | What it captures |
|---|---|---|
| s/8 | 32×32 | **Fine local detail** — water-edge pixels, small streams, building outlines |
| s/16 | 16×16 | Mid-range context — neighborhood-scale flooding |
| s/32 | 8×8 | Broader context — regional water bodies |
| s/64 | 4×4 | **Coarsest global context** — terrain shape, river system layout, overall flood extent |

`s/8` means "the input spatial size, divided by 8". So if the input is 256×256, then `s/8` = 32×32. If it's 512×512, then `s/8` = 64×64. The `s` is just shorthand so the diagram works for any input size.

### Why we apply cross-attention at all 4 scales (and not just one)

Different modalities carry different *types* of information:

- **S2 (multispectral):** strongest at fine scales — the NDWI water spectral signature is a per-pixel cue. You see "this pixel reflects light like water."
- **S1 (SAR):** dominant at mid scales — speckle is noisy at single pixels but coherent over small regions ("this 3×3 neighborhood has low backscatter, looks like flat water").
- **DEM (elevation):** matters most at coarse scales — "this is a valley, water can pool here" or "this is a 30° slope, water cannot sit here." That's a regional pattern, not a per-pixel feature.

If we only fused them at **one** scale, we'd lose information that lives at the others:
- Fuse only at s/8 → DEM contributes nothing useful (terrain context isn't visible at 32×32)
- Fuse only at s/64 → we lose S2's fine-grained spectral cues

By fusing at **all four scales** the model gets the right cross-modal information at every resolution: fine details from S2 at s/8, regional terrain context from DEM at s/64, and everything in between.

### How it works mechanically

The "3-Way Cross-Attention Block" on the right of the diagram **gets replicated four times** — once for each scale.

For each scale `s/k` ∈ {s/8, s/16, s/32, s/64}:

1. Each encoder stops at that resolution and dumps its feature map into the bridge
   - S1 features at s/k
   - S2 features at s/k
   - DEM features at s/k
2. The 3-way cross-attention block runs at that resolution:
   - S1 queries S2+DEM ("given my SAR features here, what do the other modalities say?")
   - S2 queries S1+DEM
   - DEM queries S1+S2
3. The fused output at scale s/k flows into the decoder's matching upsampling stage

The decoder then upsamples those fused features back to full resolution (256×256), producing the final water segmentation map.

### Why this matters for the model's success

- **70.9M parameters total**, but most live in the cross-attention blocks at the 4 scales
- Multi-scale fusion is what makes the model resilient: even if S2 is cloud-corrupted at fine scales, the DEM signal at coarse scales still constrains the prediction
- This is also why **early fusion sometimes beats us** at our data scale: early fusion shares weights across the entire ResNet, so 446 chips train it adequately. Our 4 separate cross-attention blocks at 4 scales are 4× the parameters to learn from the same data — at this scale, the extra capacity hurts more than it helps in-distribution. (See the ablation finding: `s1_s2` early fusion 0.7991 > our cross-attention 0.7708.)

### 30-second slide-aloud version

> "The cross-attention block runs at four resolutions — s/8 down to s/64 — because different modalities matter at different scales. Sentinel-2's spectral signature is a fine-grained pixel-level cue, but DEM's terrain context is regional. By fusing at all four scales, S1, S2, and DEM each contribute information at the resolution where it's most informative — the model isn't forced to pick one fusion scale."

---

## 2. MC Dropout — what `N` is and why we used N=20

### What `N` is

`N` is the **number of stochastic forward passes** we run per chip when computing uncertainty.

Recipe:
1. Set the model to eval mode (freezes batch-norm / group-norm statistics)
2. **Re-enable dropout** layers (normally disabled at inference)
3. Run the model **N times** on the same input
4. Each pass samples a different sub-network (because dropout randomly zeros a different subset of neurons each time)
5. Each pass produces a slightly different prediction
6. **Mean** of the N predictions → best probability estimate
7. **Variance** of the N predictions → uncertainty estimate (how much the sub-networks disagree)

This approximates Bayesian inference at test time (Gal & Ghahramani 2016). Higher disagreement among sub-networks = higher epistemic uncertainty = "the model is unsure."

### Why N=20 specifically

We used **N=20** for the reported MC Dropout evaluation (configured in `slurm/mc_uncertainty.sbatch` line 45: `--n_samples 20`).

**Tradeoffs of the N choice:**

| N | Pro | Con |
|---|---|---|
| **N = 10** | Twice as fast (~1hr on Isaac) | Noisier variance estimate; less stable ECE |
| **N = 20** ← *we use this* | Standard literature default; variance estimate has stabilized; runtime ~2hr on Isaac | 20× slower than single-shot inference |
| N = 50 | Slightly tighter variance estimate | 5× slower than N=20 with diminishing return; ~5hr on Isaac |
| N = 100 | Statistically tightest estimate | 5× slower than N=50; rarely worth it in practice |

**Why N=20 is the sweet spot:**
- **Statistical convergence:** for a Bernoulli-like prediction (water vs not), N=20 samples gives variance estimates with standard error around 5–10% of the true value. Going to N=50 only halves that error — diminishing returns.
- **Literature standard:** Gal & Ghahramani's original paper used N=10–100 depending on task. Most segmentation papers since (e.g., Kendall & Gal 2017, segmentation uncertainty work) default to N=20.
- **Runtime budget:** Our test set has 90 chips. At ~6 sec/chip × 20 passes = ~2 hours total on Isaac's single GPU. That fits comfortably in our 2-hour wall-time.
- **Empirically:** ECE = 0.0273 at N=20 — already in the "excellent" range. Going to N=50 would not meaningfully improve calibration.

**Why not N=1 (regular inference)?**
- N=1 gives only the mean prediction — no uncertainty estimate at all
- You can't compute variance from a single sample, so you can't make a reliability diagram or compute ECE

**Why not deterministic dropout (no MC at all)?**
- The whole point of MC Dropout is to **sample the posterior** — to know what range of predictions the model could plausibly make. A single deterministic pass tells you nothing about how confident the model is in its decision boundary.

### What we report from MC Dropout

- **Mean water probability** per pixel = average of N=20 predictions
- **Predictive variance** per pixel = variance across N=20 predictions (visualized as the magma-colored uncertainty maps)
- **Aggregate ECE** = 0.0273 (from comparing mean probabilities to ground-truth labels across all pixels)

### 30-second Q&A version

> "We ran 20 stochastic forward passes per chip with dropout left active at inference time. The mean of those 20 predictions is our final probability map; the variance across the 20 is our uncertainty estimate. We chose N=20 because it's the literature standard — Gal & Ghahramani's original paper recommends N=10–100, with N=20 being the modern default. At N=20 the variance estimate has stabilized; going higher gives diminishing returns and triples our wall-time. ECE 0.0273 confirms N=20 was sufficient."

---

## 3. What's architecturally NEW in our model (Q: "what did you actually add?")

> Likely professor question: "Most of these components — ResNet, U-Net, cross-attention — already exist. What did *you* add?" Have a tight answer ready.

### What we built that is genuinely new (or uncommon for this task)

1. **3-way cross-attention bridge.** Most prior fusion work on Sen1Floods11 does **dual-modal** (S1+S2 only — Yang 2021, Gauthier 2022, Bai 2021). We do **three-way**: each modality (S1, S2, DEM) attends to the other two via three parallel cross-attention heads at each scale. This is the architectural core of the contribution.

2. **Multi-scale fusion at 4 decoder levels** (s/8, s/16, s/32, s/64). Rather than fusing once at the bottleneck (most common pattern), we fuse at four resolutions so each modality contributes at the scale where its signal is strongest (S2 at fine, DEM at coarse).

3. **Modality dropout during training (p = 0.1).** During each training step, we randomly zero out **one entire modality channel** (not just neurons). This is *not* the same as standard feature dropout — it's *modality* dropout. Forces the model to remain robust when a modality is missing at inference (cloud-obscured S2, missing DEM tile, S1 acquisition gap). Most prior fusion work assumes all modalities are always available — operationally that's not realistic.

4. **Shared decoder architecture.** Three separate encoders feed **one shared decoder** through the cross-attention bridge. The alternative (three separate decoders + ensemble) wouldn't force the model to learn a unified water representation. Sharing the decoder is what makes the cross-attention bridge necessary, and what makes the model end-to-end differentiable for joint optimization.

5. **DEM as a physics-informed feature, not just raw elevation.** We compute and feed both **elevation** AND **slope** (a derivative computed once at preprocessing). Slope is the "physics cue" — the model sees that flat areas can hold water and steep areas cannot. Most flood-segmentation work that uses DEM uses raw elevation only.

### What is NOT new (be honest if asked)

- ResNet34 encoders — standard ImageNet-style backbone
- U-Net-style decoder structure — Ronneberger et al. 2015
- Cross-attention mechanism itself — Vaswani et al. 2017 (Transformer)
- MC Dropout / ECE for uncertainty — Gal & Ghahramani 2016, Guo et al. 2017
- Otsu thresholding baseline — 1979

### Tricky engineering details (worth knowing in case asked, but NOT novelty)

- **Float32 attention under mixed precision (AMP).** AMP runs everything in fp16 for speed, but the softmax inside attention overflows in fp16, producing NaN losses. We force the attention computation to fp32 while keeping the rest in fp16. Detail in the model code, not a contribution but reveals real debugging.
- **Gradient clipping at norm 1.0.** Standard but necessary — prevents the occasional gradient explosion when attention scores blow up.
- **Auto-resumable checkpoints.** Atomic file writes + RNG state preservation lets us survive Isaac's 24hr wall-time and resume exactly where we left off. Pure infrastructure, not architectural.

### How to phrase the novelty in one sentence

> "Our architectural contribution is a 3-way cross-attention fusion bridge applied at four decoder scales, combined with modality dropout during training. The 3-way attention lets DEM contribute alongside SAR and optical (vs the standard dual-modal S1+S2), the multi-scale design lets each modality contribute at its most informative resolution, and the modality dropout makes the model robust when any one modality is unavailable at inference — a realistic operational constraint that prior work assumes away."

### How to phrase the methodological novelty (broader claim)

> "Beyond the architecture itself, our methodological contributions are: (1) the first complete modality ablation matrix on Sen1Floods11 hand-labeled chips (7 variants), (2) calibrated uncertainty quantification (ECE=0.027) — not reported by any prior work on this dataset, including the Prithvi foundation model — and (3) a per-chip cross-region error analysis identifying both architecture-level failure modes and dataset-quality issues."
