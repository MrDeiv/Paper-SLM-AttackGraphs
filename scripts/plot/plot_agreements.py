#!/usr/bin/env python3
"""
Plot annotation agreement metrics as JPEG figures.

This script creates:
1. Raw agreement by metric
2. Cohen's kappa vs Gwet's AC1/AC2
3. Overview plot with raw, Cohen's kappa, and Gwet's AC1/AC2

The script is intentionally simple and self-contained.
"""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


def main():
    # ---------------------------------------------------------------------
    # Input data
    # ---------------------------------------------------------------------
    metrics = [
        "Condition Validity",
        "Evidence Support",
        "Directness",
        "Meaningfulness",
        "Atomicity",
        "Type Validity",
        "Non-redundancy",
    ]

    agreements = [293, 627, 380, 686, 788, 585, 637]
    total = 860

    raw = [0.34, 0.72, 0.44, 0.79, 0.91, 0.68, 0.74]
    cohen = [0.06, 0.29, 0.027, 0.12, 0.35, 0.36, 0.28]
    gwet = [-0.12, 0.87, 0.19, 0.74, 0.90, 0.53, 0.78]

    # ---------------------------------------------------------------------
    # Output directory
    # ---------------------------------------------------------------------
    out_dir = Path("figures")
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------------
    # Figure 1: Raw agreement
    # ---------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 5))
    y = np.arange(len(metrics))

    ax.barh(y, raw)
    ax.set_yticks(y)
    ax.set_yticklabels(metrics)
    ax.set_xlabel("Raw agreement")
    ax.set_title("Raw agreement by annotation metric")
    ax.set_xlim(0, 1)

    # Add labels with both raw agreement and agreement counts
    for i, (r, a) in enumerate(zip(raw, agreements)):
        ax.text(r + 0.01, i, f"{r:.2f} ({a}/{total})", va="center")

    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(out_dir / "agreement_raw.jpg", dpi=300, format="jpg")
    plt.close()

    # ---------------------------------------------------------------------
    # Figure 2: Chance-corrected agreement
    # ---------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(metrics))
    width = 0.35

    ax.bar(x - width / 2, cohen, width, label="Cohen's Kappa")
    ax.bar(x + width / 2, gwet, width, label="Gwet's AC1/AC2")

    ax.set_xticks(x)
    ax.set_xticklabels(metrics, rotation=30, ha="right")
    ax.set_ylabel("Agreement")
    ax.set_title("Chance-corrected agreement by annotation metric")
    ax.axhline(0, linewidth=1)
    ax.legend()

    # Add value labels
    for i, v in enumerate(cohen):
        ax.text(i - width / 2, v + (0.02 if v >= 0 else -0.06), f"{v:.2f}", ha="center")
    for i, v in enumerate(gwet):
        ax.text(i + width / 2, v + (0.02 if v >= 0 else -0.06), f"{v:.2f}", ha="center")

    plt.tight_layout()
    plt.savefig(out_dir / "agreement_corrected.jpg", dpi=300, format="jpg")
    plt.close()

    # ---------------------------------------------------------------------
    # Figure 3: Overview plot
    # ---------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(metrics))

    ax.plot(x, raw, marker="o", linestyle="-", label="Raw")
    ax.plot(x, cohen, marker="s", linestyle="--", label="Cohen's Kappa")
    ax.plot(x, gwet, marker="^", linestyle=":", label="Gwet's AC1/AC2")

    ax.set_xticks(x)
    ax.set_xticklabels(metrics, rotation=30, ha="right")
    ax.set_ylabel("Agreement")
    ax.set_title("Agreement overview by annotation metric")
    ax.axhline(0, linewidth=1)
    ax.set_ylim(-0.2, 1.0)
    ax.legend()

    plt.tight_layout()
    plt.savefig(out_dir / "agreement_overview.jpg", dpi=300, format="jpg")
    plt.close()

    print("Saved figures:")
    print(out_dir / "agreement_raw.jpg")
    print(out_dir / "agreement_corrected.jpg")
    print(out_dir / "agreement_overview.jpg")


if __name__ == "__main__":
    main()