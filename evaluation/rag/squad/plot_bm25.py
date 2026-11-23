#!/usr/bin/env python3
"""
Plot RAG retrieval metrics from a CSV produced by the SQuAD evaluation scripts.

The script creates three separated PDF figures:

1. Precision, recall, F1, F2 @ k
2. Hit and MRR @ k
3. Retrieved context tokens @ k

Example:
    python evaluation/rag/squad/plot_retrieval_csv.py \
      --csv evaluation/rag/results/squad_chunks/bm25_by_k.csv \
      --out evaluation/rag/results/squad_chunks/figures \
      --name bm25

For Windows PowerShell:
    python evaluation/rag/squad/plot_retrieval_csv.py `
      --csv evaluation/rag/results/squad_chunks/bm25_by_k.csv `
      --out evaluation/rag/results/squad_chunks/figures `
      --name bm25
"""

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
    "context_tokens_mean",
    "context_tokens_median",
    "context_tokens_p95",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot retrieval metrics from a CSV file.")
    parser.add_argument("--csv", required=True, help="Input CSV file, e.g., bm25_by_k.csv")
    parser.add_argument("--out", required=True, help="Output directory for PDF figures")
    parser.add_argument(
        "--name",
        default=None,
        help="Filename prefix for output PDFs. Defaults to the input CSV stem.",
    )
    parser.add_argument(
        "--title",
        default=None,
        help="Optional plot title prefix. Defaults to --name.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    csv_path = Path(args.csv)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    name = args.name or csv_path.stem
    title_prefix = args.title or name

    df = pd.read_csv(csv_path)
    validate_columns(df, csv_path)

    # Ensure k is numeric and ordered correctly.
    df["k"] = pd.to_numeric(df["k"], errors="coerce")
    df = df.dropna(subset=["k"]).copy()
    df["k"] = df["k"].astype(int)
    df = df.sort_values("k")

    # If the CSV contains multiple retriever configurations, for example hybrid
    # weights, each configuration is plotted as a separate line group.
    group_cols = infer_group_columns(df)

    plot_ragas_prf(
        df=df,
        group_cols=group_cols,
        title=f"{title_prefix}: Metrics @ k",
        path=out_dir / f"{name}_ragas_prf_by_k.pdf",
    )

    plot_hit_mrr(
        df=df,
        group_cols=group_cols,
        title=f"{title_prefix}: Ranking metrics @ k",
        path=out_dir / f"{name}_ranking_by_k.pdf",
    )

    plot_context_tokens(
        df=df,
        group_cols=group_cols,
        title=f"{title_prefix}: Retrieved context tokens @ k",
        path=out_dir / f"{name}_context_tokens_by_k.pdf",
    )

    print(f"Saved figures to: {out_dir}")
    print(f"- {out_dir / f'{name}_ragas_prf_by_k.pdf'}")
    print(f"- {out_dir / f'{name}_ranking_by_k.pdf'}")
    print(f"- {out_dir / f'{name}_context_tokens_by_k.pdf'}")


def validate_columns(df: pd.DataFrame, csv_path: Path) -> None:
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(
            f"CSV file {csv_path} is missing required columns: {missing}\n"
            f"Available columns: {list(df.columns)}"
        )


def infer_group_columns(df: pd.DataFrame) -> list[str]:
    """Return columns that identify different retrieval configurations.

    For BM25/vector files, this is usually just `retriever`.
    For hybrid files, this may include bm25_weight, vector_weight, bm25_k,
    vector_k, or final_k.
    """
    candidates = [
        "retriever",
        "bm25_weight",
        "vector_weight",
        "bm25_k",
        "vector_k",
        "final_k",
        "max_k_retrieved",
    ]

    group_cols: list[str] = []
    for col in candidates:
        if col in df.columns and df[col].nunique(dropna=False) > 1:
            group_cols.append(col)

    # If no varying grouping column exists, use a single artificial group.
    return group_cols


def group_label(group_key: object, group_cols: list[str]) -> str:
    if not group_cols:
        return ""

    if not isinstance(group_key, tuple):
        group_key = (group_key,)

    parts = []
    for col, value in zip(group_cols, group_key):
        if pd.isna(value):
            continue
        if isinstance(value, float):
            parts.append(f"{col}={value:.2f}")
        else:
            parts.append(f"{col}={value}")
    return ", ".join(parts)


def iter_groups(df: pd.DataFrame, group_cols: list[str]):
    if not group_cols:
        yield "", df
    else:
        for group_key, group in df.groupby(group_cols, dropna=False):
            yield group_label(group_key, group_cols), group.sort_values("k")


def setup_matplotlib() -> None:
    # Keep PDFs editable in vector-editing tools and readable in LaTeX workflows.
    plt.rcParams.update(
        {
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "figure.dpi": 150,
        }
    )


def save_plot(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close()


def plot_ragas_prf(df: pd.DataFrame, group_cols: list[str], title: str, path: Path) -> None:
    """Plot Ragas precision, recall, F1, and F2 as functions of k."""
    setup_matplotlib()
    plt.figure(figsize=(6.2, 3.8))

    metrics = [
        ("ragas_precision", "Precision"),
        ("ragas_recall", "Recall"),
        ("ragas_f1", "F1"),
        ("ragas_f2", "F2"),
    ]

    for label, group in iter_groups(df, group_cols):
        for col, metric_name in metrics:
            line_label = metric_name if not label else f"{metric_name} ({label})"
            plt.plot(group["k"], group[col], marker="o", label=line_label)

    plt.xlabel("k")
    plt.ylabel("Score")
    plt.title(title)
    plt.xticks(sorted(df["k"].unique()))
    plt.ylim(bottom=0)
    plt.legend(fontsize=7)
    save_plot(path)


def plot_hit_mrr(df: pd.DataFrame, group_cols: list[str], title: str, path: Path) -> None:
    """Plot Hit@k and MRR@k as ranking-oriented retrieval metrics."""
    setup_matplotlib()
    plt.figure(figsize=(6.2, 3.8))

    metrics = [
        ("hit", "Hit"),
        ("mrr", "MRR"),
    ]

    for label, group in iter_groups(df, group_cols):
        for col, metric_name in metrics:
            line_label = metric_name if not label else f"{metric_name} ({label})"
            plt.plot(group["k"], group[col], marker="o", label=line_label)

    plt.xlabel("k")
    plt.ylabel("Score")
    plt.title(title)
    plt.xticks(sorted(df["k"].unique()))
    plt.ylim(bottom=0)
    plt.legend(fontsize=7)
    save_plot(path)


def plot_context_tokens(df: pd.DataFrame, group_cols: list[str], title: str, path: Path) -> None:
    """Plot mean, median, and p95 retrieved context tokens as functions of k."""
    setup_matplotlib()
    plt.figure(figsize=(6.2, 3.8))

    metrics = [
        ("context_tokens_mean", "Mean tokens"),
        ("context_tokens_median", "Median tokens"),
        ("context_tokens_p95", "P95 tokens"),
    ]

    for label, group in iter_groups(df, group_cols):
        for col, metric_name in metrics:
            line_label = metric_name if not label else f"{metric_name} ({label})"
            plt.plot(group["k"], group[col], marker="o", label=line_label)

    plt.xlabel("k")
    plt.ylabel("Retrieved context tokens")
    plt.title(title)
    plt.xticks(sorted(df["k"].unique()))
    plt.ylim(bottom=0)
    plt.legend(fontsize=7)
    save_plot(path)


if __name__ == "__main__":
    main()