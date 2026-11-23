import pandas as pd
import numpy as np
import json

if __name__ == "__main__":
    ADJ_FILE = "annotation_round_1/adjudication/adjudications.csv"
    df = pd.read_csv(
        ADJ_FILE,
        dtype=str,
        encoding="utf-8-sig",
        keep_default_na=False,
    )
    
    # conditions
    conditions = df[df["field_type"] == "condition"].copy()
    
    # => preconditions
    preconditions = conditions[conditions["condition_type"] == "precondition"].copy()
    # => postconditions
    postconditions = conditions[conditions["condition_type"] == "postcondition"].copy()
    
    # descriptions
    descriptions = df[df["field_type"] == "description"].copy()
    
    stats = {}
    
    stats["n_conditions"] = len(conditions)
    stats["preconditions"] = {}
    stats["preconditions"]["n"] = len(preconditions)
    stats["postconditions"] = {}
    stats["postconditions"]["n"] = len(postconditions)
    stats["descriptions"] = {}
    stats["descriptions"]["n"] = len(descriptions)
    
    # compute agreements for condition rows only
    METRICS = [
        "evidence_support",
        "meaningfulness",
        "atomicity",
        "type_validity",
        "non_redundancy"
    ]
    
    WEIGHTED_METRICS = [
        "evidence_support",
        "type_validity",
        "non_redundancy"
    ]

    for metric in METRICS:
        label_a = f"{metric}_A"
        label_b = f"{metric}_B"
        
        # preconditions
        a = preconditions[label_a].str.strip()
        b = preconditions[label_b].str.strip()
        strict_raw_agreement = (a == b).mean()
        stats["preconditions"][f"{metric}_strict"] = strict_raw_agreement
        
        if metric in WEIGHTED_METRICS:
            # OPTIMISTIC loose metrics: 1 and 2 are considered equivalent
            a_loose = a.replace("1", "2")
            b_loose = b.replace("1", "2")
            
            loose_raw_agreement = (a_loose == b_loose).mean()
            stats["preconditions"][f"{metric}_loose_optimistic"] = loose_raw_agreement

            # PESSIMISTIC loose metrics: 1 and 0 are considered equivalent
            a_loose = a.replace("1", "0")
            b_loose = b.replace("1", "0")
            
            loose_raw_agreement = (a_loose == b_loose).mean()
            stats["preconditions"][f"{metric}_loose_pessimistic"] = loose_raw_agreement
        
        # postconditions
        a = postconditions[label_a].str.strip()
        b = postconditions[label_b].str.strip()
        strict_raw_agreement = (a == b).mean()
        stats["postconditions"][f"{metric}_strict"] = strict_raw_agreement
        
        if metric in WEIGHTED_METRICS:
            a_loose = a.replace("1", "2")
            b_loose = b.replace("1", "2")
            
            loose_raw_agreement = (a_loose == b_loose).mean()
            stats["postconditions"][f"{metric}_loose_optimistic"] = loose_raw_agreement

            a_loose = a.replace("1", "0")
            b_loose = b.replace("1", "0")
            loose_raw_agreement = (a_loose == b_loose).mean()
            stats["postconditions"][f"{metric}_loose_pessimistic"] = loose_raw_agreement

    # in-depth analysis of evidence support of conditions
    stats["evidence_support"] = {}
    # both annotators scored evidence support as 2
    valid_evidence_support_strict = conditions[
        (conditions["evidence_support_A"] == "2") & (conditions["evidence_support_B"] == "2")
    ]
    stats['evidence_support']['n_valid_evidence_support'] = len(valid_evidence_support_strict)

    # both annotators scored evidence support as 2 or 1 (optimistic)
    valid_evidence_support_loose_optimistic = conditions[
        (conditions["evidence_support_A"].isin(["1", "2"])) & (conditions["evidence_support_B"].isin(["1", "2"]))
    ]
    stats['evidence_support']['n_valid_evidence_support_loose_optimistic'] = len(valid_evidence_support_loose_optimistic)

    # both annotators scored evidence support as 0
    failure_evidence_support_strict = conditions[
        (conditions["evidence_support_A"] == "0") & (conditions["evidence_support_B"] == "0")
    ]
    stats['evidence_support']['n_failure_evidence_support_strict'] = len(failure_evidence_support_strict)

    # both annotators scored evidence support as 0 or 1 (pessimistic)
    failure_evidence_support_loose_pessimistic = conditions[
        (conditions["evidence_support_A"].isin(["0", "1"])) & (conditions["evidence_support_B"].isin(["0", "1"]))
    ]
    stats['evidence_support']['n_failure_evidence_support_loose_pessimistic'] = len(failure_evidence_support_loose_pessimistic)


    # in-depth analysis of meaningfulness of conditions
    stats["meaningfulness"] = {}
    # both annotators scored meaningfulness as 1
    valid_meaningfulness_strict = conditions[
        (conditions["meaningfulness_A"] == "1") & (conditions["meaningfulness_B"] == "1")
    ]
    stats['meaningfulness']['n_valid_meaningfulness_strict'] = len(valid_meaningfulness_strict)

    # both annotators scored meaningfulness as 0
    failure_meaningfulness_strict = conditions[
        (conditions["meaningfulness_A"] == "0") & (conditions["meaningfulness_B"] == "0")
    ]
    stats['meaningfulness']['n_failure_meaningfulness_strict'] = len(failure_meaningfulness_strict)

    # in-depth analysis of atomicity of conditions
    stats["atomicity"] = {}
    # both annotators scored atomicity as 1
    valid_atomicity_strict = conditions[
        (conditions["atomicity_A"] == "1") & (conditions["atomicity_B"] == "1")
    ]
    stats['atomicity']['n_valid_atomicity_strict'] = len(valid_atomicity_strict)

    failure_atomicity_strict = conditions[
        (conditions["atomicity_A"] == "0") & (conditions["atomicity_B"] == "0")
    ]   
    stats['atomicity']['n_failure_atomicity_strict'] = len(failure_atomicity_strict)

    # in-depth analysis of type validity of conditions
    stats["type_validity"] = {}
    # both annotators scored type validity as 2
    valid_type_validity_strict = conditions[
        (conditions["type_validity_A"] == "2") & (conditions["type_validity_B"] == "2")
    ]
    stats['type_validity']['n_valid_type_validity_strict'] = len(valid_type_validity_strict)

    # both annotators scored type validity as 2 or 1 (optimistic)
    valid_type_validity_loose_optimistic = conditions[
        (conditions["type_validity_A"].isin(["1", "2"])) & (conditions["type_validity_B"].isin(["1", "2"]))
    ]
    stats['type_validity']['n_valid_type_validity_loose_optimistic'] = len(valid_type_validity_loose_optimistic)

    # both annotators scored type validity as 0
    failure_type_validity_strict = conditions[
        (conditions["type_validity_A"] == "0") & (conditions["type_validity_B"] == "0")
    ]
    stats['type_validity']['n_failure_type_validity_strict'] = len(failure_type_validity_strict)

    # both annotators scored type validity as 0 or 1 (pessimistic)
    failure_type_validity_loose_pessimistic = conditions[
        (conditions["type_validity_A"].isin(["0", "1"])) & (conditions["type_validity_B"].isin(["0", "1"]))
    ]
    stats['type_validity']['n_failure_type_validity_loose_pessimistic'] = len(failure_type_validity_loose_pessimistic)

    # both annotators scored type validity as 1 (mistype)
    mistype_type_validity_strict = conditions[
        (conditions["type_validity_A"] == "1") & (conditions["type_validity_B"] == "1")
    ]
    stats['type_validity']['n_mistype_type_validity_strict'] = len(mistype_type_validity_strict)

    # in-depth analysis of non-redundancy of conditions
    stats["non_redundancy"] = {}
    # both annotators scored non-redundancy as 2
    valid_non_redundancy_strict = conditions[
        (conditions["non_redundancy_A"] == "2") & (conditions["non_redundancy_B"] == "2")
    ]
    stats['non_redundancy']['n_valid_non_redundancy_strict'] = len(valid_non_redundancy_strict)

    # both annotators scored non-redundancy as 2 or 1 (optimistic)
    valid_non_redundancy_loose_optimistic = conditions[
        (conditions["non_redundancy_A"].isin(["1", "2"])) & (conditions["non_redundancy_B"].isin(["1", "2"]))
    ]
    stats['non_redundancy']['n_valid_non_redundancy_loose_optimistic'] = len(valid_non_redundancy_loose_optimistic)

    # both annotators scored non-redundancy as 0
    failure_non_redundancy_strict = conditions[
        (conditions["non_redundancy_A"] == "0") & (conditions["non_redundancy_B"] == "0")
    ]
    stats['non_redundancy']['n_failure_non_redundancy_strict'] = len(failure_non_redundancy_strict)

    # both annotators scored non-redundancy as 0 or 1 (pessimistic)
    failure_non_redundancy_loose_pessimistic = conditions[
        (conditions["non_redundancy_A"].isin(["0", "1"])) & (conditions["non_redundancy_B"].isin(["0", "1"]))
    ]
    stats['non_redundancy']['n_failure_non_redundancy_loose_pessimistic'] = len(failure_non_redundancy_loose_pessimistic)


    # in-depth analysis of evidence support of descriptions
    stats["description_evidence_support"] = {}

    # both annotators scored evidence support as 2
    valid_description_evidence_support_strict = descriptions[
        (descriptions["description_evidence_support_A"] == "2") & (descriptions["description_evidence_support_B"] == "2")
    ]
    stats['description_evidence_support']['n_valid_description_evidence_support_strict'] = len(valid_description_evidence_support_strict)

    # both annotators scored evidence support as 2 or 1 (optimistic)
    valid_description_evidence_support_loose_optimistic = descriptions[
        (descriptions["description_evidence_support_A"].isin(["1", "2"])) & (descriptions["description_evidence_support_B"].isin(["1", "2"]))
    ]
    stats['description_evidence_support']['n_valid_description_evidence_support_loose_optimistic'] = len(valid_description_evidence_support_loose_optimistic)

    # both annotators scored evidence support as 0
    failure_description_evidence_support_strict = descriptions[
        (descriptions["description_evidence_support_A"] == "0") & (descriptions["description_evidence_support_B"] == "0")
    ]
    stats['description_evidence_support']['n_failure_description_evidence_support_strict'] = len(failure_description_evidence_support_strict)

    # both annotators scored evidence support as 0 or 1 (pessimistic)
    failure_description_evidence_support_loose_pessimistic = descriptions[
        (descriptions["description_evidence_support_A"].isin(["0", "1"])) & (descriptions["description_evidence_support_B"].isin(["0", "1"]))
    ]
    stats['description_evidence_support']['n_failure_description_evidence_support_loose_pessimistic'] = len(failure_description_evidence_support_loose_pessimistic)

    # in-depth analysis of faithfulness of descriptions
    stats["description_faithfulness"] = {}

    # both annotators scored faithfulness as 2
    valid_description_faithfulness_strict = descriptions[
        (descriptions["description_faithfulness_A"] == "2") & (descriptions["description_faithfulness_B"] == "2")
    ]
    stats['description_faithfulness']['n_valid_description_faithfulness_strict'] = len(valid_description_faithfulness_strict)

    # both annotators scored faithfulness as 2 or 1 (optimistic)
    valid_description_faithfulness_loose_optimistic = descriptions[
        (descriptions["description_faithfulness_A"].isin(["1", "2"])) & (descriptions["description_faithfulness_B"].isin(["1", "2"]))
    ]
    stats['description_faithfulness']['n_valid_description_faithfulness_loose_optimistic'] = len(valid_description_faithfulness_loose_optimistic)

    # both annotators scored faithfulness as 0
    failure_description_faithfulness_strict = descriptions[
        (descriptions["description_faithfulness_A"] == "0") & (descriptions["description_faithfulness_B"] == "0")
    ]
    stats['description_faithfulness']['n_failure_description_faithfulness_strict'] = len(failure_description_faithfulness_strict)

    # both annotators scored faithfulness as 0 or 1 (pessimistic)
    failure_description_faithfulness_loose_pessimistic = descriptions[
        (descriptions["description_faithfulness_A"].isin(["0", "1"])) & (descriptions["description_faithfulness_B"].isin(["0", "1"]))
    ]
    stats['description_faithfulness']['n_failure_description_faithfulness_loose_pessimistic'] = len(failure_description_faithfulness_loose_pessimistic)

    # in-depth analysis of specificity of descriptions
    stats["description_specificity"] = {}

    # both annotators scored specificity as 2
    valid_description_specificity_strict = descriptions[
        (descriptions["description_specificity_A"] == "2") & (descriptions["description_specificity_B"] == "2")
    ]
    stats['description_specificity']['n_valid_description_specificity_strict'] = len(valid_description_specificity_strict)

    # both annotators scored specificity as 2 or 1 (optimistic)
    valid_description_specificity_loose_optimistic = descriptions[
        (descriptions["description_specificity_A"].isin(["1", "2"])) & (descriptions["description_specificity_B"].isin(["1", "2"]))
    ]
    stats['description_specificity']['n_valid_description_specificity_loose_optimistic'] = len(valid_description_specificity_loose_optimistic)

    # both annotators scored specificity as 0
    failure_description_specificity_strict = descriptions[
        (descriptions["description_specificity_A"] == "0") & (descriptions["description_specificity_B"] == "0")
    ]
    stats['description_specificity']['n_failure_description_specificity_strict'] = len(failure_description_specificity_strict)

    # both annotators scored specificity as 0 or 1 (pessimistic)
    failure_description_specificity_loose_pessimistic = descriptions[
        (descriptions["description_specificity_A"].isin(["0", "1"])) & (descriptions["description_specificity_B"].isin(["0", "1"]))
    ]
    stats['description_specificity']['n_failure_description_specificity_loose_pessimistic'] = len(failure_description_specificity_loose_pessimistic)

    # in-depth analysis of unsupported claims of descriptions
    stats["description_unsupported_claims"] = {}

    # both annotators scored unsupported claims as 2
    valid_description_unsupported_claims_strict = descriptions[
        (descriptions["unsupported_claims_A"] == "2") & (descriptions["unsupported_claims_B"] == "2")
    ]
    stats['description_unsupported_claims']['n_valid_description_unsupported_claims_strict'] = len(valid_description_unsupported_claims_strict)

    # both annotators scored unsupported claims as 2 or 1 (optimistic)
    valid_description_unsupported_claims_loose_optimistic = descriptions[
        (descriptions["unsupported_claims_A"].isin(["1", "2"])) & (descriptions["unsupported_claims_B"].isin(["1", "2"]))
    ]
    stats['description_unsupported_claims']['n_valid_description_unsupported_claims_loose_optimistic'] = len(valid_description_unsupported_claims_loose_optimistic)

    # both annotators scored unsupported claims as 0
    failure_description_unsupported_claims_strict = descriptions[
        (descriptions["unsupported_claims_A"] == "0") & (descriptions["unsupported_claims_B"] == "0")
    ]
    stats['description_unsupported_claims']['n_failure_description_unsupported_claims_strict'] = len(failure_description_unsupported_claims_strict)

    # both annotators scored unsupported claims as 0 or 1 (pessimistic)
    failure_description_unsupported_claims_loose_pessimistic = descriptions[
        (descriptions["unsupported_claims_A"].isin(["0", "1"])) & (descriptions["unsupported_claims_B"].isin(["0", "1"]))
    ]
    stats['description_unsupported_claims']['n_failure_description_unsupported_claims_loose_pessimistic'] = len(failure_description_unsupported_claims_loose_pessimistic)
    
    with open("annotation_round_1/adjudication/agreement_stats_detail.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=4)
        

        
        
    
    
    
    
    
    