from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


REQUIRED_COLUMNS = [
    "k",
    "bm25_weight",
    "vector_weight",
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

OPTIONAL_COLUMNS = [
    "ndcg",
    "bm25_k",
    "vector_k",
    "num_queries",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot hybrid retrieval evaluation results.")
    parser.add_argument("--csv", required=True, help="Input hybrid_by_k.csv file.")
    parser.add_argument("--out", required=True, help="Output directory for PDF figures.")
    parser.add_argument("--selected-k", type=int, default=50, help="Selected final hybrid k. Default: 50.")
    parser.add_argument("--bm25-weight", type=float, default=0.25, help="Selected BM25 weight. Default: 0.25.")
    parser.add_argument("--name", default="hybrid", help="Prefix for output PDF filenames. Default: hybrid.")
    parser.add_argument(
        "--title",
        default="Hybrid retrieval",
        help="Title prefix used in the figures. Default: Hybrid retrieval.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    csv_path = Path(args.csv)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)
    validate_columns(df, csv_path)
    df = normalize_dataframe(df)

    selected_k = nearest_available_value(df["k"], args.selected_k)
    selected_weight = nearest_available_value(df["bm25_weight"], args.bm25_weight)

    if selected_k != args.selected_k:
        print(f"Requested k={args.selected_k}, but nearest available k is {selected_k}.")
    if abs(selected_weight - args.bm25_weight) > 1e-9:
        print(f"Requested BM25 weight={args.bm25_weight}, but nearest available weight is {selected_weight}.")

    setup_matplotlib()

    plot_prf_by_weight(
        df=df,
        selected_k=selected_k,
        selected_weight=selected_weight,
        title=args.title,
        path=out_dir / f"{args.name}_prf_by_weight_at_k{selected_k}.pdf",
    )

    plot_ranking_by_weight(
        df=df,
        selected_k=selected_k,
        selected_weight=selected_weight,
        title=args.title,
        path=out_dir / f"{args.name}_ranking_by_weight_at_k{selected_k}.pdf",
    )

    plot_prf_by_k_fixed_weight(
        df=df,
        selected_k=selected_k,
        selected_weight=selected_weight,
        title=args.title,
        path=out_dir / f"{args.name}_prf_by_k_bm25w{format_weight(selected_weight)}.pdf",
    )

    plot_ranking_by_k_fixed_weight(
        df=df,
        selected_k=selected_k,
        selected_weight=selected_weight,
        title=args.title,
        path=out_dir / f"{args.name}_ranking_by_k_bm25w{format_weight(selected_weight)}.pdf",
    )

    plot_k_choice_tradeoff(
        df=df,
        selected_k=selected_k,
        selected_weight=selected_weight,
        title=args.title,
        path=out_dir / f"{args.name}_k_choice_tradeoff_k{selected_k}_bm25w{format_weight(selected_weight)}.pdf",
    )

    write_choice_summary(
        df=df,
        selected_k=selected_k,
        selected_weight=selected_weight,
        path=out_dir / f"{args.name}_choice_summary.csv",
    )

    print(f"Saved figures to {out_dir}")
    print(f"- {args.name}_prf_by_weight_at_k{selected_k}.pdf")
    print(f"- {args.name}_ranking_by_weight_at_k{selected_k}.pdf")
    print(f"- {args.name}_prf_by_k_bm25w{format_weight(selected_weight)}.pdf")
    print(f"- {args.name}_ranking_by_k_bm25w{format_weight(selected_weight)}.pdf")
    print(f"- {args.name}_k_choice_tradeoff_k{selected_k}_bm25w{format_weight(selected_weight)}.pdf")
    print(f"- {args.name}_choice_summary.csv")


def validate_columns(df: pd.DataFrame, csv_path: Path) -> None:
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(
            f"{csv_path} is missing required columns: {missing}\n"
            f"Available columns: {list(df.columns)}"
        )


def normalize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = REQUIRED_COLUMNS + [col for col in OPTIONAL_COLUMNS if col in df.columns]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["k", "bm25_weight"]).copy()
    df["k"] = df["k"].astype(int)
    df = df.sort_values(["bm25_weight", "k"]).reset_index(drop=True)
    return df


def nearest_available_value(series: pd.Series, requested: float) -> float:
    values = sorted(series.dropna().unique())
    if not values:
        raise ValueError("No available values found.")
    return min(values, key=lambda value: abs(float(value) - float(requested)))


def setup_matplotlib() -> None:
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


def selected_label(selected_weight: float) -> str:
    return f"BM25 weight={selected_weight:.2f}, vector weight={1.0 - selected_weight:.2f}"


def filter_at_k(df: pd.DataFrame, selected_k: int) -> pd.DataFrame:
    out = df[df["k"] == selected_k].copy()
    if out.empty:
        raise ValueError(f"No rows found for k={selected_k}. Available k values: {sorted(df['k'].unique())}")
    return out.sort_values("bm25_weight")


def filter_at_weight(df: pd.DataFrame, selected_weight: float) -> pd.DataFrame:
    out = df[(df["bm25_weight"] - selected_weight).abs() < 1e-9].copy()
    if out.empty:
        available = sorted(df["bm25_weight"].dropna().unique())
        raise ValueError(f"No rows found for BM25 weight={selected_weight}. Available weights: {available}")
    return out.sort_values("k")


def plot_prf_by_weight(
    *,
    df: pd.DataFrame,
    selected_k: int,
    selected_weight: float,
    title: str,
    path: Path,
) -> None:
    """Plot Ragas precision/recall/F1/F2 as BM25/vector weight varies at fixed k."""
    data = filter_at_k(df, selected_k)

    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    ax.plot(data["bm25_weight"], data["ragas_precision"], marker="o", linestyle=":", label="Precision")
    ax.plot(data["bm25_weight"], data["ragas_recall"], marker="^", linestyle="--", label="Recall")
    ax.plot(data["bm25_weight"], data["ragas_f1"], marker="s", linestyle="-", label="F1")
    ax.plot(data["bm25_weight"], data["ragas_f2"], marker="D", linestyle="-.", label="F2")

    selected_row = get_row(data, k=selected_k, weight=selected_weight)
    ax.axvline(selected_weight, linestyle="--", linewidth=1.2)
    ax.scatter([selected_weight], [selected_row["ragas_f2"]], s=65, zorder=5)

    ax.set_xlabel("BM25 weight")
    ax.set_ylabel("Score")
    ax.set_title(f"{title}: Metrics by weight at k={selected_k}")
    ax.set_xticks(sorted(data["bm25_weight"].unique()))
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=8)

    save_plot(path)


def plot_ranking_by_weight(
    *,
    df: pd.DataFrame,
    selected_k: int,
    selected_weight: float,
    title: str,
    path: Path,
) -> None:
    """Plot Hit and MRR as BM25/vector weight varies at fixed k."""
    data = filter_at_k(df, selected_k)

    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    ax.plot(data["bm25_weight"], data["hit"], marker="o", linestyle="-", label="Hit")
    ax.plot(data["bm25_weight"], data["mrr"], marker="s", linestyle="--", label="MRR")

    selected_row = get_row(data, k=selected_k, weight=selected_weight)
    ax.axvline(selected_weight, linestyle="--", linewidth=1.2)
    #ax.scatter([selected_weight], [selected_row["hit"]], s=65, zorder=5)

    ax.set_xlabel("BM25 weight")
    ax.set_ylabel("Score")
    ax.set_title(f"{title}: ranking metrics by weight at k={selected_k}")
    ax.set_xticks(sorted(data["bm25_weight"].unique()))
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=8)

    save_plot(path)


def plot_prf_by_k_fixed_weight(
    *,
    df: pd.DataFrame,
    selected_k: int,
    selected_weight: float,
    title: str,
    path: Path,
) -> None:
    """Plot Ragas precision/recall/F1/F2 as k varies at fixed BM25/vector weight."""
    data = filter_at_weight(df, selected_weight)

    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    ax.plot(data["k"], data["ragas_precision"], marker="o", linestyle=":", label="Precision")
    ax.plot(data["k"], data["ragas_recall"], marker="^", linestyle="--", label="Recall")
    ax.plot(data["k"], data["ragas_f1"], marker="s", linestyle="-", label="F1")
    ax.plot(data["k"], data["ragas_f2"], marker="D", linestyle="-.", label="F2")

    selected_row = get_row(data, k=selected_k, weight=selected_weight)
    ax.axvline(selected_k, linestyle="--", linewidth=1.2)
    ax.scatter([selected_k], [selected_row["ragas_f2"]], s=65, zorder=5)

    ax.set_xlabel("Final hybrid k")
    ax.set_ylabel("Score")
    ax.set_title(f"{title}: Metrics by k")
    ax.set_xticks(sorted(data["k"].unique()))
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=8)

    save_plot(path)


def plot_ranking_by_k_fixed_weight(
    *,
    df: pd.DataFrame,
    selected_k: int,
    selected_weight: float,
    title: str,
    path: Path,
) -> None:
    """Plot Hit and MRR as k varies at fixed BM25/vector weight."""
    data = filter_at_weight(df, selected_weight)

    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    ax.plot(data["k"], data["hit"], marker="o", linestyle="-", label="Hit")
    ax.plot(data["k"], data["mrr"], marker="s", linestyle="--", label="MRR")

    selected_row = get_row(data, k=selected_k, weight=selected_weight)
    ax.axvline(selected_k, linestyle="--", linewidth=1.2)
    ax.scatter([selected_k], [selected_row["hit"]], s=65, zorder=5)

    ax.set_xlabel("Final hybrid k")
    ax.set_ylabel("Score")
    ax.set_title(f"{title}: ranking metrics by k")
    ax.set_xticks(sorted(data["k"].unique()))
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=8)

    save_plot(path)


def plot_k_choice_tradeoff(
    *,
    df: pd.DataFrame,
    selected_k: int,
    selected_weight: float,
    title: str,
    path: Path,
) -> None:

    data = filter_at_weight(df, selected_weight)

    _, ax = plt.subplots(figsize=(6.2, 3.8))
    ax.plot(data["context_tokens_mean"], data["ragas_f2"], marker="o", linestyle="-")

    for _, row in data.iterrows():
        ax.annotate(
            f"k={int(row['k'])}",
            (row["context_tokens_mean"], row["ragas_f2"]),
            xytext=(4, 4),
            textcoords="offset points",
        )

    selected_row = get_row(data, k=selected_k, weight=selected_weight)
    max_k = int(data["k"].max())
    max_row = data[data["k"] == max_k].iloc[0]

    ax.scatter(
        [selected_row["context_tokens_mean"]],
        [selected_row["ragas_f2"]],
        s=80,
        zorder=5,
    )
    
    ax.set_xlabel("Mean retrieved context tokens")
    ax.set_ylabel("F2")
    ax.set_title(f"{title}: k-choice trade-off\n{selected_label(selected_weight)}")
    ax.set_ylim(bottom=0)

    save_plot(path)


def get_row(df: pd.DataFrame, *, k: int, weight: float) -> pd.Series:
    rows = df[(df["k"] == k) & ((df["bm25_weight"] - weight).abs() < 1e-9)]
    if rows.empty:
        raise ValueError(f"No row found for k={k}, bm25_weight={weight}.")
    return rows.iloc[0]


def write_choice_summary(
    *,
    df: pd.DataFrame,
    selected_k: int,
    selected_weight: float,
    path: Path,
) -> None:
    data = filter_at_weight(df, selected_weight)

    selected_row = get_row(data, k=selected_k, weight=selected_weight)
    max_k = int(data["k"].max())
    max_row = data[data["k"] == max_k].iloc[0]

    summary = {
        "selected_k": selected_k,
        "bm25_weight": selected_weight,
        "vector_weight": 1.0 - selected_weight,
        "selected_ragas_precision": selected_row["ragas_precision"],
        "selected_ragas_recall": selected_row["ragas_recall"],
        "selected_ragas_f1": selected_row["ragas_f1"],
        "selected_ragas_f2": selected_row["ragas_f2"],
        "selected_hit": selected_row["hit"],
        "selected_mrr": selected_row["mrr"],
        "selected_ndcg": selected_row["ndcg"] if "ndcg" in selected_row else None,
        "selected_context_tokens_mean": selected_row["context_tokens_mean"],
        "max_k": max_k,
        "max_k_ragas_f2": max_row["ragas_f2"],
        "max_k_context_tokens_mean": max_row["context_tokens_mean"],
        "delta_f2_max_minus_selected": max_row["ragas_f2"] - selected_row["ragas_f2"],
        "delta_tokens_max_minus_selected": max_row["context_tokens_mean"] - selected_row["context_tokens_mean"],
        "selected_f2_retained_vs_max_k": selected_row["ragas_f2"] / max_row["ragas_f2"],
        "selected_token_ratio_vs_max_k": selected_row["context_tokens_mean"] / max_row["context_tokens_mean"],
    }

    pd.DataFrame([summary]).to_csv(path, index=False)


def format_weight(weight: float) -> str:
    return str(float(weight)).replace(".", "p")


if __name__ == "__main__":
    main()