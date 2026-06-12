#!/usr/bin/env python3
"""
Classification Evaluation Script

Evaluates the performance of route classification using Precision, Recall, and F1-Score.
Compares route_intended (gold standard) against route_orch (predicted labels).
"""

import pandas as pd
from sklearn.metrics import (
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
    accuracy_score,
)
import sys


def evaluate_classification(csv_path: str):
    """
    Load CSV file and evaluate classification metrics.

    Args:
        csv_path: Path to the CSV file containing route_intended and route_orch columns
    """
    # Load the data
    print(f"Loading data from: {csv_path}")
    df = pd.read_csv(csv_path)

    # Validate required columns exist
    required_columns = ['route_intended', 'route_orch']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        print(f"Error: Missing required columns: {missing_columns}")
        sys.exit(1)

    # Check for any missing values
    if df['route_intended'].isnull().any() or df['route_orch'].isnull().any():
        print("Warning: Found null values in the data. They will be excluded from evaluation.")
        df = df.dropna(subset=['route_intended', 'route_orch'])

    y_true = df['route_intended']
    y_pred = df['route_orch']

    print(f"\nTotal samples: {len(y_true)}")
    print(f"\nLabel distribution in gold standard (route_intended):")
    print(y_true.value_counts().sort_index())
    print(f"\nLabel distribution in predictions (route_orch):")
    print(y_pred.value_counts().sort_index())

    # Get all unique labels from both columns to ensure consistent order
    all_labels = sorted(set(y_true.unique()) | set(y_pred.unique()))
    print(f"\nLabels found: {all_labels}")

    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=all_labels, average=None
    )

    # Calculate macro, micro, and weighted averages
    precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=all_labels, average='micro'
    )
    _, _, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=all_labels, average='macro'
    )
    _, _, f1_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=all_labels, average='weighted'
    )

    # Print results
    print("\n" + "="*80)
    print("CLASSIFICATION EVALUATION RESULTS")
    print("="*80)

    print(f"\nOverall Accuracy: {accuracy:.4f}\n")

    # Per-class metrics
    print("Per-Class Metrics:")
    print("-" * 80)
    print(f"{'Label':<20} {'Precision':<15} {'Recall':<15} {'F1-Score':<15} {'Support':<10}")
    print("-" * 80)

    for i, label in enumerate(all_labels):
        print(f"{label:<20} {precision[i]:<15.4f} {recall[i]:<15.4f} {f1[i]:<15.4f} {int(support[i]):<10}")

    print("-" * 80)
    print(f"{'Macro Average':<20} {precision.mean():<15.4f} {recall.mean():<15.4f} {f1_macro:<15.4f} {int(support.sum()):<10}")
    print(f"{'Micro Average':<20} {precision_micro:<15.4f} {recall_micro:<15.4f} {f1_micro:<15.4f} {int(support.sum()):<10}")
    print(f"{'Weighted Average':<20} {(precision * support / support.sum()).sum():<15.4f} {(recall * support / support.sum()).sum():<15.4f} {f1_weighted:<15.4f} {int(support.sum()):<10}")

    # Print classification report
    print("\n" + "="*80)
    print("DETAILED CLASSIFICATION REPORT")
    print("="*80 + "\n")
    print(classification_report(y_true, y_pred, labels=all_labels, digits=4))

    # Print confusion matrix
    print("\n" + "="*80)
    print("CONFUSION MATRIX")
    print("="*80)
    cm = confusion_matrix(y_true, y_pred, labels=all_labels)

    # Format confusion matrix nicely
    print(f"\n{'':>20}", end='')
    for label in all_labels:
        print(f"{label:>15}", end='')
    print()
    print("-" * (20 + 15 * len(all_labels)))

    for i, label in enumerate(all_labels):
        print(f"{label:>20}", end='')
        for j in range(len(all_labels)):
            print(f"{cm[i, j]:>15}", end='')
        print()


    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'support': support,
        'labels': all_labels,
        'precision_micro': precision_micro,
        'recall_micro': recall_micro,
        'f1_micro': f1_micro,
    }


if __name__ == "__main__":
    # Default path
    csv_file = "/final_data/all_results_v2_domains.csv"

    # Allow command-line override
    if len(sys.argv) > 1:
        csv_file = sys.argv[1]

    evaluate_classification(csv_file)

