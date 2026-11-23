import pandas as pd
import numpy as np
import json

if __name__ == "__main__":
    DATA = "annotation_round_1/adjudication/adjudications.csv"

    df = pd.read_csv(DATA)
    results = {}

    # get conditions
    conditions = df[df["field_type"] == "condition"].copy()

    # count the number of conditions where both annotators gave 2 to condition_validity and evidence_support
    results["valid_and_grounded"] = {}
    strict_count = 0
    for index, row in conditions.iterrows():
        if row["condition_validity_A"] == 2 and row["condition_validity_B"] == 2 and row["evidence_support_A"] == 2 and row["evidence_support_B"] == 2:
            strict_count += 1
    results["valid_and_grounded"]["strict_count"] = strict_count

    # loose count optimistic: 2==1
    loose_count_optimistic = 0
    for index, row in conditions.iterrows():
        if (row["condition_validity_A"] == 2 or row["condition_validity_A"] == 1) and (row["condition_validity_B"] == 2 or row["condition_validity_B"] == 1) and (row["evidence_support_A"] == 2 or row["evidence_support_A"] == 1) and (row["evidence_support_B"] == 2 or row["evidence_support_B"] == 1):
            loose_count_optimistic += 1
    results["valid_and_grounded"]["loose_count_optimistic"] = loose_count_optimistic

    # valid evidence support
    count_valid_evidence_support_loose = 0
    for index, row in conditions.iterrows():
        if (row["evidence_support_A"] == 2 or row["evidence_support_A"] == 1) and (row["evidence_support_B"] == 2 or row["evidence_support_B"] == 1):
            count_valid_evidence_support_loose += 1
    results["count_valid_evidence_support_loose"] = count_valid_evidence_support_loose

    # valid evidence support strict
    count_valid_evidence_support = 0
    for index, row in conditions.iterrows():
        if row["evidence_support_A"] == 2 and row["evidence_support_B"] == 2:
            count_valid_evidence_support += 1
    results["count_valid_evidence_support"] = count_valid_evidence_support

    with open("annotation_round_1/adjudication/qualitative_results.json", "w") as f:
        json.dump(results, f, indent=4)
