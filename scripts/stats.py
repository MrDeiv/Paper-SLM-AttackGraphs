import pandas as pd
import numpy as np
import json
from matplotlib import pyplot as plt

if __name__ == "__main__":
    stats = {}
    ADJ_FILE = "annotation_round_1/adjudication/adjudications.csv"
    df = pd.read_csv(
        ADJ_FILE,
        dtype=str,
        encoding="utf-8-sig",
        keep_default_na=False,
    )
    
    # compute agreements for condition rows only
    CONDITION_METRICS = [
        "condition_validity",
        "evidence_support",
        "directness",
        "meaningfulness",
        "atomicity",
        "type_validity",
        "non_redundancy"
    ]
    
    WEIGHTED_METRICS = [
        "condition_validity",
        "evidence_support",
        "directness",
        "type_validity",
        "non_redundancy"
    ]
    
    NON_WEIGHTED_METRICS = [
        "meaningfulness",
        "atomicity"
    ]
    
    # sum the scores in each condition row and compute the average total score for each annotator
    conditions = df[df["field_type"] == "condition"].copy()
    
    scores_a = 0
    scores_b = 0
    lowest_row_score_a = 12
    lowest_row_score_b = 12
    highest_row_score_a = 0
    highest_row_score_b = 0
    
    for condition in conditions.itertuples(index=False):
        a_scores = [
            getattr(condition, f"{metric}_A") for metric in CONDITION_METRICS
        ]
        b_scores = [
            getattr(condition, f"{metric}_B") for metric in CONDITION_METRICS
        ]
        
        a_total = sum(int(score) for score in a_scores)
        b_total = sum(int(score) for score in b_scores)
        
        scores_a += a_total
        scores_b += b_total
        
        if a_total < lowest_row_score_a:
            lowest_row_score_a = a_total
        
        if b_total < lowest_row_score_b:
            lowest_row_score_b = b_total
            
        if a_total > highest_row_score_a:
            highest_row_score_a = a_total
        
        if b_total > highest_row_score_b:
            highest_row_score_b = b_total
    
    scores_a = scores_a / len(conditions)
    scores_b = scores_b / len(conditions)
    
    std_a = np.std([sum(int(getattr(condition, f"{metric}_A")) for metric in CONDITION_METRICS) for condition in conditions.itertuples(index=False)])
    std_b = np.std([sum(int(getattr(condition, f"{metric}_B")) for metric in CONDITION_METRICS) for condition in conditions.itertuples(index=False)])
    
    stats["annotator_a"] = {
        "average_total_score": scores_a,
        "std_dev_total_score": std_a,
        "lowest_row_score": lowest_row_score_a,
        "highest_row_score": highest_row_score_a
    }
    stats["annotator_b"] = {
        "average_total_score": scores_b,
        "std_dev_total_score": std_b,
        "lowest_row_score": lowest_row_score_b,
        "highest_row_score": highest_row_score_b
    }
    
    # avg and std dev for each metric
    for metric in CONDITION_METRICS:
        a_scores = conditions[f"{metric}_A"].astype(int)
        b_scores = conditions[f"{metric}_B"].astype(int)
        
        stats[metric] = {
            "annotator_a": {
                "average_score": a_scores.mean(),
                "std_dev_score": a_scores.std()
            },
            "annotator_b": {
                "average_score": b_scores.mean(),
                "std_dev_score": b_scores.std()
            },
            "avg_score_difference": np.abs(a_scores.mean() - b_scores.mean()),
        }
    
    with open("annotation_round_1/adjudication/stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=4)
        
    # plot line graph of average scores for each metric with error bars for std dev
    metrics = CONDITION_METRICS
    
    plt.figure(figsize=(10, 6))
    plt.errorbar(metrics, [stats[metric]["annotator_a"]["average_score"] for metric in metrics],
                 yerr=[stats[metric]["annotator_a"]["std_dev_score"] for metric in metrics],
                 label="Annotator A", fmt='-o', capsize=5)
    plt.errorbar(metrics, [stats[metric]["annotator_b"]["average_score"] for metric in metrics],
                 yerr=[stats[metric]["annotator_b"]["std_dev_score"] for metric in metrics],
                 label="Annotator B", fmt='-o', capsize=5)
    
    plt.title("Average Scores for Each Metric with Standard Deviation")
    plt.xlabel("Metrics")
    plt.ylabel("Average Score")
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig("annotation_round_1/adjudication/average_scores_with_std_dev.png")
    
    
    # plot line graph of average scores for each metric without error bars
    plt.figure(figsize=(10, 6))
    plt.plot(metrics, [stats[metric]["annotator_a"]["average_score"] for metric in metrics],
             label="Annotator A", marker='o')
    plt.plot(metrics, [stats[metric]["annotator_b"]["average_score"] for metric in metrics],
             label="Annotator B", marker='o')
    plt.title("Average Scores for Each Metric")
    plt.xlabel("Metrics")
    plt.ylabel("Average Score")
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig("annotation_round_1/adjudication/average_scores.png")

    # plot the total scores for each row for both annotators. On x axis the row index, on y axis the total score. Use different colors for each annotator and add a legend.
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(conditions)), [sum(int(getattr(condition, f"{metric}_A")) for metric in CONDITION_METRICS) for condition in conditions.itertuples(index=False)],
             label="Annotator A", marker='o')
    plt.plot(range(len(conditions)), [sum(int(getattr(condition, f"{metric}_B")) for metric in CONDITION_METRICS) for condition in conditions.itertuples(index=False)],
                label="Annotator B", marker='o')
    plt.title("Total Scores for Each Row")
    plt.xlabel("Row Index")
    plt.ylabel("Total Score")
    plt.xticks(range(len(conditions)))
    plt.legend()
    plt.tight_layout()
    plt.savefig("annotation_round_1/adjudication/total_scores_per_row.png")
    
    