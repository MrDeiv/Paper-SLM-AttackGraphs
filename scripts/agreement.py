import pandas as pd
import numpy as np
from sklearn.metrics import cohen_kappa_score
import json

def compute_raw_agreement(a, b):
    """Compute raw agreement between two annotators."""
    return (a == b).mean()

def compute_agreement(a, b, weighted=False):
    """Compute agreement between two annotators."""
    if weighted:
        #if set(a.unique()) != {0, 1, 2} or set(b.unique()) != {0, 1, 2}:
            #raise ValueError("Weighted agreement requires labels to be in {0, 1, 2}.")
        return cohen_kappa_score(a, b, labels=[0, 1, 2], weights="quadratic")
    else:
        if set(a.unique()) != {0, 1} or set(b.unique()) != {0, 1}:
            raise ValueError("Non-weighted agreement requires labels to be in {0, 1}.")
        return cohen_kappa_score(a, b, labels=[0, 1])
    
def compute_agreement_loose(a, b):
    """Compute agreement between two annotators, treating 1 and 2 as equivalent."""
    return cohen_kappa_score(a, b, labels=[0, 2])

def gwet_ac1(y_a, y_b):
    """
    Compute Gwet's AC1 for two annotators with binary labels (0, 1).

    Both annotators must have evaluated the same items
    in the same order.
    """

    a = np.asarray(y_a, dtype=int)
    b = np.asarray(y_b, dtype=int)

    if len(a) != len(b):
        raise ValueError("Annotators must have the same number of labels.")

    if len(a) == 0:
        raise ValueError("No annotations provided.")

    if not np.isin(a, [0, 1]).all() or not np.isin(b, [0, 1]).all():
        raise ValueError("Gwet's AC1 implementation requires binary labels 0 and 1.")

    # Observed agreement
    po = np.mean(a == b)

    # Proportion of label 1 across both annotators
    p = (np.sum(a == 1) + np.sum(b == 1)) / (2 * len(a))

    # Gwet's AC1 chance-agreement term
    pe = 2 * p * (1 - p)

    return (po - pe) / (1 - pe)

def gwet_ac2(a, b):
    a = np.asarray(a, dtype=int)
    b = np.asarray(b, dtype=int)

    if len(a) != len(b) or len(a) == 0:
        raise ValueError("Invalid annotation lengths")

    categories = np.array([0, 1, 2])
    q = len(categories)

    if not np.isin(a, categories).all() or not np.isin(b, categories).all():
        raise ValueError("Labels must belong to {0, 1, 2}")

    # Quadratic agreement weights.
    W = 1 - (
        (categories[:, None] - categories[None, :]) ** 2
        / (q - 1) ** 2
    )

    # Observed weighted agreement.
    po = np.mean(W[a, b])

    # Pooled marginal proportions across both annotators.
    p = np.array([
        (np.sum(a == c) + np.sum(b == c)) / (2 * len(a))
        for c in categories
    ])

    # Gwet's expected agreement.
    pe = W.sum() * np.sum(p * (1 - p)) / (q * (q - 1))

    return (po - pe) / (1 - pe)

if __name__ == "__main__":
    results = {}
    
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
    
    conditions = df[df["field_type"] == "condition"].copy()
    ids = conditions["random_order"].str.strip()
    n_conditions = len(conditions)
    results["n_conditions"] = n_conditions
    
    for metric in CONDITION_METRICS:
        print(f"Computing agreement for {metric}...")
        
        label_a = f"{metric}_A"
        label_b = f"{metric}_B"
        
        a = conditions[label_a].str.strip()
        b = conditions[label_b].str.strip()
        
        a = a.astype(int)
        b = b.astype(int)

        # Raw agreement
        raw_agreement = compute_raw_agreement(a, b)       
        kappa = compute_agreement(a, b, weighted=(metric in WEIGHTED_METRICS))
        
        # verdicts
        verdict = "invalid"
        if kappa < 0.2:
            verdict = "Poor"
        elif kappa < 0.4:
            verdict = "Fair"
        elif kappa < 0.6:
            verdict = "Moderate"
        elif kappa < 0.8:
            verdict = "Substantial"
        else:
            verdict = "Almost perfect/Perfect"
            
        a = a.astype(int).to_numpy()
        b = b.astype(int).to_numpy()
        
        if metric in NON_WEIGHTED_METRICS:
            ac = gwet_ac1(a, b)
        else:
            ac = gwet_ac2(a, b)
        
        # check for major and minor disagreements
        if metric in WEIGHTED_METRICS:
            major_disagreement = np.sum(np.abs(a - b) == 2)
            minor_disagreement = np.sum(np.abs(a - b) == 1)
        else:
            major_disagreement = np.sum(a != b)
            minor_disagreement = 0
        
        if metric in WEIGHTED_METRICS:
            # OPTIMISTIC: transform 1 in 2 for loose metrics
            a_opt = np.where(a == 1, 2, a)
            b_opt = np.where(b == 1, 2, b)
            a_merged = pd.Series(a_opt)
            b_merged = pd.Series(b_opt)
    
            k_loose_opt = compute_agreement_loose(a_merged, b_merged)
            ac_loose_opt = gwet_ac2(a_merged, b_merged)
            raw_agreement_loose_opt = compute_raw_agreement(a_merged, b_merged)

            a_pess = np.where(a == 1, 0, a)
            b_pess = np.where(b == 1, 0, b)
            a_merged = pd.Series(a_pess)
            b_merged = pd.Series(b_pess)
            k_loose_pess = compute_agreement_loose(a_merged, b_merged)
            ac_loose_pess = gwet_ac2(a_merged, b_merged)
            raw_agreement_loose_pess = compute_raw_agreement(a_merged, b_merged)
        else:
            k_loose = kappa  # For non-weighted metrics, k_loose is the same as kappa   
            ac_loose = ac  # For non-weighted metrics, ac_loose is the same as ac
            raw_agreement_loose = raw_agreement  # For non-weighted metrics, raw_agreement_loose is the same as raw_agreement
        
        # verdict for k_loose
        if k_loose_opt < 0.2:
            verdict_loose_opt = "Poor"
        elif k_loose_opt < 0.4:
            verdict_loose_opt = "Fair"
        elif k_loose_opt < 0.6:
            verdict_loose_opt = "Moderate"
        elif k_loose_opt < 0.8:
            verdict_loose_opt = "Substantial"
        else:
            verdict_loose_opt = "Almost perfect/Perfect"

        if k_loose_pess < 0.2:
            verdict_loose_pess = "Poor"
        elif k_loose_pess < 0.4:
            verdict_loose_pess = "Fair"
        elif k_loose_pess < 0.6:
            verdict_loose_pess = "Moderate"
        elif k_loose_pess < 0.8:
            verdict_loose_pess = "Substantial"
        else:
            verdict_loose_pess = "Almost perfect/Perfect"
        
        
        results[metric] = {
            "raw_agreement": raw_agreement,
            "kappa_strict": kappa,
            "k_strict_verdict": verdict,
            "gwet_ac_strict": ac,
            "major_disagreement": int(major_disagreement),
            "minor_disagreement": int(minor_disagreement),
            "raw_agreement_loose_optimistic": raw_agreement_loose_opt,
            "kappa_loose_optimistic": k_loose_opt,
            "k_loose_optimistic_verdict": verdict_loose_opt,
            "gwet_ac_loose_optimistic": ac_loose_opt,
            "raw_agreement_loose_pess": raw_agreement_loose_pess,
            "kappa_loose_pessimistic": k_loose_pess,
            "k_loose_pessimistic_verdict": verdict_loose_pess,
            "gwet_ac_loose_pessimistic": ac_loose_pess
        }
        
        with open(f"annotation_round_1/adjudication/crosstabs.txt", "a", encoding="utf-8") as f:
            f.write(f"\n\nCrosstab for {metric}:\n")
            f.write(pd.crosstab(a, b, margins=True).to_string())
            f.write("\n")
    
    
    # descriptions
    stats = {}
    # descriptions
    descriptions = df[df["field_type"] == "description"].copy()
    stats["n"] = len(descriptions)
        
    METRICS = [
        "description_faithfulness",
        "description_evidence_support",
        "description_specificity",
        "unsupported_claims",
    ]
        
    for metric in METRICS:
        print(f"Computing agreement for {metric}...")
        stats[f"{metric}"] = {}
        
        label_a = f"{metric}_A"
        label_b = f"{metric}_B"
            
        a = descriptions[label_a].str.strip()
        b = descriptions[label_b].str.strip()
        
        a = a.astype(int)
        b = b.astype(int)
        
        major_disagreement = np.sum(np.abs(a - b) == 2)
        minor_disagreement = np.sum(np.abs(a - b) == 1)
        
        stats[f"{metric}"]["major_disagreement"] = int(major_disagreement)
        stats[f"{metric}"]["minor_disagreement"] = int(minor_disagreement)
            
        strict_raw_agreement = (a == b).mean()
        stats[f"{metric}"]["strict_raw_agreement"] = strict_raw_agreement
        
        # compute kappa and gwet's AC2 for descriptions
        kappa = compute_agreement(a, b, weighted=True)
        k_ac = gwet_ac2(a, b)
        
        stats[f"{metric}"]["kappa_strict"] = kappa
        stats[f"{metric}"]["gwet_ac_strict"] = k_ac
            
        # OPTIMISTIC loose metrics: 1 and 2 are considered equivalent
        a_loose = a.replace(1, 2).copy()
        b_loose = b.replace(1, 2).copy()
        
        kappa_loose = compute_agreement_loose(a_loose, b_loose)
        
        k_ac_loose = gwet_ac2(a_loose, b_loose)
        raw_agreement_loose = (a_loose == b_loose).mean()
        
        stats[f"{metric}"]["kappa_loose_optimistic"] = kappa_loose
        stats[f"{metric}"]["gwet_ac_loose_optimistic"] = k_ac_loose
        stats[f"{metric}"]["raw_agreement_loose_optimistic"] = raw_agreement_loose

        # PESSIMISTIC loose metrics: 1 and 0 are considered equivalent
        a_loose = a.replace(1, 0).copy()
        b_loose = b.replace(1, 0).copy()
        
        kappa_loose = compute_agreement_loose(a_loose, b_loose)
        
        k_ac_loose = gwet_ac2(a_loose, b_loose)
        raw_agreement_loose = (a_loose == b_loose).mean()
        
        stats[f"{metric}"]["kappa_loose_pessimistic"] = kappa_loose
        stats[f"{metric}"]["gwet_ac_loose_pessimistic"] = k_ac_loose
        stats[f"{metric}"]["raw_agreement_loose_pessimistic"] = raw_agreement_loose

    
    SAVE_PATH = "annotation_round_1/adjudication/agreement_conditions.json"
    with open(SAVE_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)
        
    SAVE_PATH = "annotation_round_1/adjudication/agreement_descriptions.json"
    with open(SAVE_PATH, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=4)