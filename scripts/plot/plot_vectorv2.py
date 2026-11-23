#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


REQUIRED_COLUMNS = [
    "k",
    "ragas_precision",
    "ragas_recall",
    "ragas_f1",
    "ragas_f2",
    "hit",
    "mrr",
    "ndcg",
    "context_tokens_mean",
    "context_tokens_median",
    "context_tokens_p95",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot retrieval metrics and highlight the selected k.")
    parser.add_argument("--csv", required=True, help="Input CSV file, e.g., bm25_by_k.csv")
    parser.add_argument("--out", required=True, help="Output directory for PDF figures")
    parser.add_argument("--name", default=None, help="Filename prefix. Defaults to the input CSV stem.")
    parser.add_argument("--title", default=None, help="Title prefix. Defaults to --name.")
    parser.add_argument("--target-k", type=int, required=True, help="Selected k to highlight, e.g., 100")
    parser.add_argument(
        "--reference-k",
        type=int,
        default=None,
        help="Optional reference k for comparison, e.g., 200.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    csv_path = Path(args.csv)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    name = args.name or csv_path.stem
    title = args.title or name

    df = pd.read_csv(csv_path)
    validate_columns(df, csv_path)

    df["k"] = pd.to_numeric(df["k"], errors="coerce")
    df = df.dropna(subset=["k"]).copy()
    df["k"] = df["k"].astype(int)
    df = df.sort_values("k").reset_index(drop=True)

    if args.target_k not in set(df["k"]):
        raise ValueError(f"target-k={args.target_k} is not present in the CSV. Available k values: {list(df['k'])}")

    setup_matplotlib()

    plot_ragas_metrics(
        df=df,
        target_k=args.target_k,
        title=f"{title}",
        path=out_dir / f"{name}_ragas_prf_highlight_k{args.target_k}.pdf",
    )

    plot_ranking_metrics(
        df=df,
        target_k=args.target_k,
        title=f"{title}: ranking metrics @ k",
        path=out_dir / f"{name}_ranking_highlight_k{args.target_k}.pdf",
    )

    plot_context_tokens(
        df=df,
        target_k=args.target_k,
        title=f"{title}: retrieved context tokens @ k",
        path=out_dir / f"{name}_context_tokens_highlight_k{args.target_k}.pdf",
    )

    plot_f2_vs_tokens(
        df=df,
        target_k=args.target_k,
        title=f"{title}: Ragas F2 vs context budget",
        path=out_dir / f"{name}_f2_vs_context_tokens_highlight_k{args.target_k}.pdf",
    )

    plot_marginal_gains(
        df=df,
        target_k=args.target_k,
        title=f"{title}: marginal gain between k values",
        path=out_dir / f"{name}_marginal_gain_highlight_k{args.target_k}.pdf",
    )

    summary = build_choice_summary(df, target_k=args.target_k, reference_k=args.reference_k)
    summary_path = out_dir / f"{name}_choice_summary_k{args.target_k}.csv"
    pd.DataFrame([summary]).to_csv(summary_path, index=False)

    print(f"Saved figures to: {out_dir}")
    print(f"- {out_dir / f'{name}_ragas_prf_highlight_k{args.target_k}.pdf'}")
    print(f"- {out_dir / f'{name}_ranking_highlight_k{args.target_k}.pdf'}")
    print(f"- {out_dir / f'{name}_context_tokens_highlight_k{args.target_k}.pdf'}")
    print(f"- {out_dir / f'{name}_f2_vs_context_tokens_highlight_k{args.target_k}.pdf'}")
    print(f"- {out_dir / f'{name}_marginal_gain_highlight_k{args.target_k}.pdf'}")
    print(f"- {summary_path}")

    print("\nChoice summary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")


def validate_columns(df: pd.DataFrame, csv_path: Path) -> None:
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(
            f"CSV file {csv_path} is missing required columns: {missing}\n"
            f"Available columns: {list(df.columns)}"
        )


def setup_matplotlib() -> None:
    # Keep text editable in vector-based PDF workflows.
    plt.rcParams.update(
        {
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "figure.dpi": 150,
        }
    )


def get_row(df: pd.DataFrame, k: int) -> pd.Series:
    rows = df[df["k"] == k]
    if rows.empty:
        raise ValueError(f"k={k} not found. Available k values: {list(df['k'])}")
    return rows.iloc[0]


def annotate_target_k(ax, df: pd.DataFrame, target_k: int, y_col: str, label: str | None = None) -> None:
    row = get_row(df, target_k)
    y = float(row[y_col])

    ax.axvline(target_k, linestyle="--", linewidth=1.2)
    ax.scatter([target_k], [y], s=55, zorder=5)

    text = label or f"k={target_k}\n{y_col}={y:.3f}"
    ax.annotate(
        text,
        xy=(target_k, y),
        xytext=(8, 10),
        textcoords="offset points",
        fontsize=8,
        bbox={"boxstyle": "round,pad=0.25", "alpha": 0.15},
    )


def save_plot(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close()


def plot_ragas_metrics(df: pd.DataFrame, target_k: int, title: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.2, 3.8))

    ax.plot(df["k"], df["ragas_precision"], marker="o", linestyle=":", label="Precision")
    ax.plot(df["k"], df["ragas_recall"], marker="^", linestyle="--", label="Recall")
    ax.plot(df["k"], df["ragas_f1"], marker="s", linestyle="-", label="F1")
    ax.plot(df["k"], df["ragas_f2"], marker="D", linestyle="-.", label="F2")

    annotate_target_k(ax, df, target_k, "ragas_f2", label=f"selected k={target_k}\nF2={get_row(df, target_k)['ragas_f2']:.3f}")

    ax.set_xlabel("k")
    ax.set_ylabel("Score")
    ax.set_title(title)
    ax.set_xticks(list(df["k"]))
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=8)

    save_plot(path)


def plot_ranking_metrics(df: pd.DataFrame, target_k: int, title: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.2, 3.8))

    ax.plot(df["k"], df["hit"], marker="o", linestyle="-", label="Hit")
    ax.plot(df["k"], df["mrr"], marker="s", linestyle="--", label="MRR")
    ax.plot(df["k"], df["ndcg"], marker="^", linestyle=":", label="nDCG")

    annotate_target_k(ax, df, target_k, "hit", label=f"selected k={target_k}\nHit={get_row(df, target_k)['hit']:.3f}")

    ax.set_xlabel("k")
    ax.set_ylabel("Score")
    ax.set_title(title)
    ax.set_xticks(list(df["k"]))
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=8)

    save_plot(path)


def plot_context_tokens(df: pd.DataFrame, target_k: int, title: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.2, 3.8))

    ax.plot(df["k"], df["context_tokens_mean"], marker="o", linestyle="-", label="Mean tokens")
    ax.plot(df["k"], df["context_tokens_median"], marker="s", linestyle="--", label="Median tokens")
    ax.plot(df["k"], df["context_tokens_p95"], marker="^", linestyle=":", label="P95 tokens")

    row = get_row(df, target_k)
    annotate_target_k(
        ax,
        df,
        target_k,
        "context_tokens_mean",
        label=f"selected k={target_k}\nmean={row['context_tokens_mean']:.0f} tokens",
    )

    ax.set_xlabel("k")
    ax.set_ylabel("Retrieved context tokens")
    ax.set_title(title)
    ax.set_xticks(list(df["k"]))
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=8)

    save_plot(path)


def plot_f2_vs_tokens(df: pd.DataFrame, target_k: int, title: str, path: Path) -> None:
    """Plot the core trade-off: retrieval quality versus context budget."""
    fig, ax = plt.subplots(figsize=(6.0, 3.8))

    ax.plot(df["context_tokens_mean"], df["ragas_f2"], marker="o", linestyle="-")

    for _, row in df.iterrows():
        k = int(row["k"])
        ax.annotate(
            str(k),
            xy=(row["context_tokens_mean"], row["ragas_f2"]),
            xytext=(5, 4),
            textcoords="offset points",
            fontsize=8,
        )

    target = get_row(df, target_k)
    ax.scatter([target["context_tokens_mean"]], [target["ragas_f2"]], s=70, zorder=5)
    ax.annotate(
        f"selected k={target_k}\nF2={target['ragas_f2']:.3f}\nmean={target['context_tokens_mean']:.0f} tokens",
        xy=(target["context_tokens_mean"], target["ragas_f2"]),
        xytext=(12, 12),
        textcoords="offset points",
        fontsize=8,
        bbox={"boxstyle": "round,pad=0.25", "alpha": 0.15},
    )

    ax.set_xlabel("Mean retrieved context tokens")
    ax.set_ylabel("Ragas F2")
    ax.set_title(title)
    ax.set_ylim(bottom=0)

    save_plot(path)


def plot_marginal_gains(df: pd.DataFrame, target_k: int, title: str, path: Path) -> None:
    """Show diminishing returns between adjacent k values.

    This plot is useful for justifying k=100 over k=200: it shows how much
    additional Ragas F2 and recall are gained when moving to the next k.
    """
    deltas = df[["k", "ragas_recall", "ragas_f2", "context_tokens_mean"]].copy()
    deltas["prev_k"] = deltas["k"].shift(1)
    deltas["delta_recall"] = deltas["ragas_recall"].diff()
    deltas["delta_f2"] = deltas["ragas_f2"].diff()
    deltas["delta_tokens"] = deltas["context_tokens_mean"].diff()
    deltas = deltas.dropna().copy()

    labels = [f"{int(row.prev_k)}→{int(row.k)}" for _, row in deltas.iterrows()]
    x = range(len(deltas))

    fig, ax1 = plt.subplots(figsize=(6.6, 3.8))

    ax1.plot(x, deltas["delta_f2"], marker="o", linestyle="-", label="Δ F2")
    ax1.plot(x, deltas["delta_recall"], marker="s", linestyle="--", label="Δ Recall")
    ax1.set_ylabel("Metric gain")
    ax1.set_xlabel("k transition")
    ax1.set_xticks(list(x))
    ax1.set_xticklabels(labels, rotation=30, ha="right")
    ax1.set_title(title)

    ax2 = ax1.twinx()
    ax2.plot(x, deltas["delta_tokens"], marker="^", linestyle=":", label="Δ mean tokens")
    ax2.set_ylabel("Additional mean tokens")

    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, fontsize=8, loc="best")

    save_plot(path)


def build_choice_summary(df: pd.DataFrame, target_k: int, reference_k: int | None) -> dict:
    target = get_row(df, target_k)

    summary = {
        "target_k": target_k,
        "target_ragas_precision": float(target["ragas_precision"]),
        "target_ragas_recall": float(target["ragas_recall"]),
        "target_ragas_f1": float(target["ragas_f1"]),
        "target_ragas_f2": float(target["ragas_f2"]),
        "target_hit": float(target["hit"]),
        "target_mrr": float(target["mrr"]),
        "target_ndcg": float(target["ndcg"]),
        "target_context_tokens_mean": float(target["context_tokens_mean"]),
        "target_context_tokens_median": float(target["context_tokens_median"]),
        "target_context_tokens_p95": float(target["context_tokens_p95"]),
    }

    if reference_k is not None and reference_k in set(df["k"]):
        ref = get_row(df, reference_k)
        summary.update(
            {
                "reference_k": reference_k,
                "reference_ragas_f2": float(ref["ragas_f2"]),
                "reference_ragas_recall": float(ref["ragas_recall"]),
                "reference_context_tokens_mean": float(ref["context_tokens_mean"]),
                "delta_f2_reference_minus_target": float(ref["ragas_f2"] - target["ragas_f2"]),
                "delta_recall_reference_minus_target": float(ref["ragas_recall"] - target["ragas_recall"]),
                "delta_context_tokens_reference_minus_target": float(
                    ref["context_tokens_mean"] - target["context_tokens_mean"]
                ),
                "f2_retained_vs_reference": float(target["ragas_f2"] / ref["ragas_f2"]) if ref["ragas_f2"] else None,
                "token_ratio_target_vs_reference": float(target["context_tokens_mean"] / ref["context_tokens_mean"])
                if ref["context_tokens_mean"]
                else None,
            }
        )

    return summary


if __name__ == "__main__":
    main()