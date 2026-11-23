import pandas as pd
import numpy as np
import json

if __name__ == "__main__":
    adjudication_file = "annotation_round_1/adjudication/adjudications.csv"
    
    df = pd.read_csv(adjudication_file)
    
    # get only conditions
    conditions = df[df['field_type'] == 'condition'].copy()
    
    CONDITION_METRICS = [
        "condition_validity",
        "evidence_support",
        "directness",
        "meaningfulness",
        "atomicity",
        "type_validity",
        "non_redundancy"
    ]
    
    results = {}
    for metric in CONDITION_METRICS:
        results[metric] = {}
        label_A = f"{metric}_A"
        label_B = f"{metric}_B"
        
        # both annotators give lowest score
        both_lowest_n = len(conditions[(conditions[label_A] == 0) & (conditions[label_B] == 0)])
        results[metric]['agreed_failures'] = both_lowest_n
        
    # only preconditions
    preconditions = conditions[conditions['condition_type'] == 'precondition'].copy()
    
    for metric in CONDITION_METRICS:
        label_A = f"{metric}_A"
        label_B = f"{metric}_B"
        
        # both annotators give lowest score
        both_lowest_n = len(preconditions[(preconditions[label_A] == 0) & (preconditions[label_B] == 0)])
        results[metric]['agreed_failures_preconditions'] = both_lowest_n
        
    # only postconditions
    postconditions = conditions[conditions['condition_type'] == 'postcondition'].copy()
    
    for metric in CONDITION_METRICS:
        label_A = f"{metric}_A"
        label_B = f"{metric}_B"
        
        # both annotators give lowest score
        both_lowest_n = len(postconditions[(postconditions[label_A] == 0) & (postconditions[label_B] == 0)])
        results[metric]['agreed_failures_postconditions'] = both_lowest_n
        
        
    # descriptions
    descriptions = df[df['field_type'] == 'description'].copy()
    
    METRICS = [
        "description_faithfulness",
        "description_evidence_support",
        "description_specificity",
        "unsupported_claims",
    ]
        
    for metric in METRICS:
        results[metric] = {}
        label_A = f"{metric}_A"
        label_B = f"{metric}_B"
        
        # both annotators give lowest score
        both_lowest_n = len(descriptions[(descriptions[label_A] == 0) & (descriptions[label_B] == 0)])
        results[metric]['agreed_failures'] = both_lowest_n
        
    with open("annotation_round_1/adjudication/failure_stats.json", "w") as f:
        json.dump(results, f, indent=4)
    