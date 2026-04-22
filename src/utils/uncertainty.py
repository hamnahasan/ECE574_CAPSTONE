"""MC Dropout uncertainty estimation for flood segmentation models.

SCIENTIFIC BACKGROUND:
    Standard neural networks produce a single deterministic prediction.
    They cannot express HOW CONFIDENT they are in that prediction.
    For disaster response, knowing uncertainty is as important as the
    prediction itself — a model should say "I'm unsure about this region"
    rather than confidently predicting flood in an ambiguous area.

    Monte Carlo Dropout (Gal & Ghahramani, 2016) approximates Bayesian
    inference at test time by keeping dropout layers ACTIVE and running
    the model N times. Each forward pass samples a different sub-network,
    producing slightly different predictions. The variance across predictions
    is the uncertainty estimate.

    TWO UNCERTAINTY TYPES:
    1. Epistemic (model) uncertainty — captured by MC Dropout variance.
       High where training data was sparse or the model has not seen
       similar inputs before. Can be reduced with more data.

    2. Aleatoric (data) uncertainty — inherent noise in the input.
       High where SAR speckle, cloud cover, or ambiguous boundaries exist.
       Cannot be reduced with more data.

    For flood segmentation, high uncertainty should correspond to:
    - Cloud-covered S2 pixels
    - SAR layover/shadow regions
    - Flood boundaries (partially inundated pixels)
    - Regions geographically different from training (cross-region test)

CALIBRATION:
    A well-calibrated model has predicted confidence = observed accuracy.
    Expected Calibration Error (ECE) measures how far off calibration is:
        ECE = sum_b (|B_b| / N) * |acc(B_b) - conf(B_b)|
    where bins B_b group predictions by confidence level.

    Reliability diagrams plot confidence vs accuracy — a perfectly
    calibrated model lies on the diagonal.

Usage:
    from src.utils.uncertainty import mc_predict, compute_ece, reliability_diagram

    # Run MC Dropout with N=20 forward passes
    mean_prob, uncertainty = mc_predict(model, s1, s2, dem, n_samples=20)

    # Compute ECE on the test set
    ece = compute_ece(all_probs, all_labels, n_bins=15)
"""

import numpy as np
import torch
import torch.nn.functional as F


def enable_dropout(model):
    """Set all Dropout/Dropout2d layers to train mode (keeps them active).

    By default, model.eval() disables dropout. MC Dropout requires dropout
    to remain ACTIVE at inference to sample from the approximate posterior.
    We selectively re-enable only dropout layers, keeping BN/GN in eval mode.
    """
    for m in model.modules():
        if isinstance(m, (torch.nn.Dropout, torch.nn.Dropout2d)):
            m.train()


@torch.no_grad()
def mc_predict(model, s1, s2=None, dem=None, n_samples=20, device=None):
    """Run N stochastic forward passes and return mean prediction + uncertainty.

    The model is set to eval() first (freezes GroupNorm statistics), then
    dropout is re-enabled for stochastic sampling. This matches the standard
    MC Dropout protocol from Gal & Ghahramani (2016).

    Args:
        model:     Trained model (FusionUNet, TriModalFusionUNet, or EarlyFusionUNet).
        s1:        (1, 2,  H, W) S1 tensor (required).
        s2:        (1, 13, H, W) S2 tensor or None.
        dem:       (1, 2,  H, W) DEM tensor or None.
        n_samples: Number of stochastic forward passes (N=20 is standard;
                   higher N = better estimate but slower inference).
        device:    Torch device. Inferred from s1 if None.

    Returns:
        mean_prob:   (H, W) mean water probability across N passes. [0, 1]
        uncertainty: (H, W) predictive uncertainty (variance of water prob).
                     High values = uncertain, low values = confident.
    """
    if device is None:
        device = next(model.parameters()).device

    # Set eval mode (freezes norm layers) then re-enable dropout
    model.eval()
    enable_dropout(model)

    prob_stack = []  # collect water probability from each pass

    for _ in range(n_samples):
        # Select correct forward pass signature based on available modalities
        if s2 is not None and dem is not None:
            # Tri-modal or dual-modal
            if dem is not None and hasattr(model, 'dem_encoder'):
                logits = model(s1.to(device), s2.to(device), dem.to(device))
            else:
                logits = model(s1.to(device), s2.to(device))
        elif s2 is not None:
            logits = model(s1.to(device), s2.to(device))
        else:
            # S1-only (FCN baseline) or early fusion (caller pre-concatenates)
            logits = model(s1.to(device))

        # Softmax to get per-class probabilities; take water class (index 1)
        prob = F.softmax(logits, dim=1)[0, 1]  # (H, W)
        prob_stack.append(prob.cpu().float())

    # Stack: (N, H, W)
    prob_stack = torch.stack(prob_stack, dim=0)

    # Mean prediction: average water probability across all N passes
    mean_prob = prob_stack.mean(dim=0)  # (H, W)

    # Predictive uncertainty: variance across passes.
    # High variance = model disagrees with itself = uncertain region.
    uncertainty = prob_stack.var(dim=0)  # (H, W)

    return mean_prob.numpy(), uncertainty.numpy()


def compute_ece(probs, labels, n_bins=15, ignore_index=255):
    """Compute Expected Calibration Error (ECE).

    ECE measures the gap between predicted confidence and actual accuracy.
    A model outputting 0.8 water probability should be correct 80% of the time.
    If it is only correct 50% of the time, it is overconfident.

    ECE = sum_b (|B_b| / N) * |accuracy(B_b) - confidence(B_b)|

    Args:
        probs:        (N,) or (H, W) flattened water probabilities [0, 1].
        labels:       (N,) or (H, W) ground truth binary labels {0, 1}.
        n_bins:       Number of confidence bins (15 is standard in literature).
        ignore_index: Label value to exclude (255 = nodata).

    Returns:
        ece:  float — Expected Calibration Error (lower is better, 0 = perfect).
        bins: dict with per-bin accuracy, confidence, and count for plotting.
    """
    probs  = np.array(probs).flatten()
    labels = np.array(labels).flatten()

    # Remove nodata pixels
    valid  = labels != ignore_index
    probs  = probs[valid]
    labels = labels[valid]

    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    bins = {"accuracy": [], "confidence": [], "count": []}

    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        in_bin = (probs >= lo) & (probs < hi)
        count  = in_bin.sum()

        if count == 0:
            bins["accuracy"].append(0.0)
            bins["confidence"].append((lo + hi) / 2)
            bins["count"].append(0)
            continue

        acc  = (labels[in_bin] == 1).mean()   # fraction correct in bin
        conf = probs[in_bin].mean()            # mean predicted probability

        # Weighted gap: bins with more samples contribute more to ECE
        ece += (count / len(probs)) * abs(acc - conf)

        bins["accuracy"].append(float(acc))
        bins["confidence"].append(float(conf))
        bins["count"].append(int(count))

    return float(ece), bins


def reliability_diagram(bins, ece, save_path=None):
    """Plot a reliability diagram (confidence vs accuracy).

    A perfectly calibrated model lies on the diagonal.
    Bars above diagonal = underconfident (predicted less than actual).
    Bars below diagonal = overconfident (predicted more than actual).

    Args:
        bins:      dict from compute_ece() with accuracy, confidence, count.
        ece:       float ECE value to display in title.
        save_path: Optional path to save the figure.

    Returns:
        matplotlib Figure.
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    n_bins = len(bins["accuracy"])
    bin_centers = [(i + 0.5) / n_bins for i in range(n_bins)]
    width = 1.0 / n_bins

    fig, ax = plt.subplots(figsize=(6, 6))

    # Perfect calibration reference line
    ax.plot([0, 1], [0, 1], "k--", linewidth=1.5, label="Perfect calibration")

    # Accuracy bars (blue) and gap to diagonal (red = overconfidence)
    for i, (acc, conf, cnt) in enumerate(
        zip(bins["accuracy"], bin_centers, bins["count"])
    ):
        if cnt == 0:
            continue
        # Blue bar = actual accuracy in bin
        ax.bar(bin_centers[i], acc, width=width * 0.9,
               color="#3182bd", alpha=0.8, align="center", edgecolor="white")
        # Red gap = overconfidence (if bar below diagonal)
        if acc < bin_centers[i]:
            ax.bar(bin_centers[i], bin_centers[i] - acc, width=width * 0.9,
                   bottom=acc, color="#d73027", alpha=0.4,
                   align="center", edgecolor="none")

    ax.set_xlabel("Confidence (predicted water probability)", fontsize=12)
    ax.set_ylabel("Accuracy (fraction correctly labeled water)", fontsize=12)
    ax.set_title(f"Reliability Diagram — ECE = {ece:.4f}", fontsize=13)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)

    blue_patch = mpatches.Patch(color="#3182bd", alpha=0.8, label="Accuracy per bin")
    red_patch  = mpatches.Patch(color="#d73027", alpha=0.4, label="Overconfidence gap")
    ax.legend(handles=[blue_patch, red_patch,
                        plt.Line2D([0],[0], color='k', linestyle='--',
                                   label="Perfect calibration")],
              fontsize=9, loc="upper left")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def evaluate_uncertainty(model, dataset, device, n_samples=20,
                          n_bins=15, save_dir=None):
    """Run MC Dropout on an entire dataset split and compute ECE.

    Args:
        model:     Trained model with dropout layers.
        dataset:   Sen1Floods11TriModal or similar dataset instance.
        device:    Torch device.
        n_samples: MC Dropout samples per chip.
        n_bins:    ECE calibration bins.
        save_dir:  Optional directory to save reliability diagram.

    Returns:
        dict with ece, mean_uncertainty, per_chip results.
    """
    from tqdm import tqdm

    all_probs  = []
    all_labels = []
    chip_uncertainties = []

    print(f"Running MC Dropout ({n_samples} samples) on {len(dataset)} chips...")
    for idx in tqdm(range(len(dataset))):
        # Handle both tri-modal and dual-modal datasets
        sample = dataset[idx]
        if len(sample) == 4:
            s1, s2, dem, label = sample
            mean_prob, uncertainty = mc_predict(
                model, s1.unsqueeze(0), s2.unsqueeze(0), dem.unsqueeze(0),
                n_samples=n_samples, device=device,
            )
        else:
            s1, s2, label = sample
            mean_prob, uncertainty = mc_predict(
                model, s1.unsqueeze(0), s2.unsqueeze(0),
                n_samples=n_samples, device=device,
            )

        lbl_np = label.numpy()
        valid  = lbl_np != 255
        all_probs.append(mean_prob[valid])
        all_labels.append(lbl_np[valid])
        chip_uncertainties.append(float(uncertainty[valid].mean()))

    all_probs  = np.concatenate(all_probs)
    all_labels = np.concatenate(all_labels)

    ece, bins = compute_ece(all_probs, all_labels, n_bins=n_bins)
    mean_unc  = float(np.mean(chip_uncertainties))

    print(f"ECE:              {ece:.4f}  (lower is better)")
    print(f"Mean uncertainty: {mean_unc:.4f} (higher = less confident overall)")

    if save_dir:
        import os; os.makedirs(save_dir, exist_ok=True)
        fig = reliability_diagram(bins, ece,
                                  save_path=f"{save_dir}/reliability_diagram.png")
        import matplotlib.pyplot as plt; plt.close(fig)

    return {
        "ece": ece,
        "mean_uncertainty": mean_unc,
        "bins": bins,
        "per_chip_uncertainty": chip_uncertainties,
    }
