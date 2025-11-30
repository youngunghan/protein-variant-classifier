import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, precision_recall_curve
import matplotlib.pyplot as plt

def analyze_scores(file_path):
    # Load data
    df = pd.read_csv(file_path)
    
    predictors = ['SCORE_A', 'SCORE_B', 'SCORE_C']
    results = {}

    print(f"Total patients: {df['Patient_ID'].nunique()}")
    print(f"Total variants: {len(df)}")
    
    # 1. Global Metrics (AUROC, AUPRC)
    print("\n--- Global Metrics ---")
    for pred in predictors:
        auroc = roc_auc_score(df['LABEL'], df[pred])
        auprc = average_precision_score(df['LABEL'], df[pred])
        results[pred] = {'AUROC': auroc, 'AUPRC': auprc}
        print(f"{pred}: AUROC = {auroc:.4f}, AUPRC = {auprc:.4f}")

    # 2. Patient-Centric Metrics (Ranking)
    print("\n--- Patient-Centric Metrics (Ranking) ---")
    
    # how often the pathogenic variant (LABEL=1) is ranked highly for each patient.    
    patient_groups = df.groupby('Patient_ID')
    
    ranking_results = {pred: {'Top-1': 0, 'Top-5': 0} for pred in predictors}
    total_patients_with_pathogenic = 0
    
    for patient_id, group in patient_groups:
        # Check if patient has any pathogenic variants
        if group['LABEL'].sum() == 0:
            continue
            
        total_patients_with_pathogenic += 1
        
        for pred in predictors:
            # Sort variants by score descending
            sorted_group = group.sort_values(by=pred, ascending=False).reset_index(drop=True)
            
            # Find indices of pathogenic variants
            pathogenic_indices = sorted_group.index[sorted_group['LABEL'] == 1].tolist()
            
            # Check if any pathogenic variant is at rank 0 (Top-1)
            if 0 in pathogenic_indices:
                ranking_results[pred]['Top-1'] += 1
                
            # Check if any pathogenic variant is in top 5
            if any(idx < 5 for idx in pathogenic_indices):
                ranking_results[pred]['Top-5'] += 1

    print(f"Patients with at least one pathogenic variant: {total_patients_with_pathogenic}")
    
    for pred in predictors:
        top1_acc = ranking_results[pred]['Top-1'] / total_patients_with_pathogenic
        top5_recall = ranking_results[pred]['Top-5'] / total_patients_with_pathogenic
        results[pred]['Top-1 Accuracy'] = top1_acc
        results[pred]['Top-5 Recall'] = top5_recall
        print(f"{pred}: Top-1 Accuracy = {top1_acc:.4f}, Top-5 Recall = {top5_recall:.4f}")

    return results

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Analyze pathogenicity prediction scores')
    parser.add_argument('file_path', type=str, help='Path to the CSV file containing scores and labels')
    args = parser.parse_args()
    
    analyze_scores(args.file_path)
