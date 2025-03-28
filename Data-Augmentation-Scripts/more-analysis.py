import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# ---- Fairness Metrics Function ----
def compute_fairness_from_accuracy(df, demographic_col='Demographic Variables',
                                   pred_col='Is Correct Original'):
    results = []
    groups = df[demographic_col].unique()

    for group in groups:
        subset = df[df[demographic_col] == group]
        correct = subset[pred_col]

        total = len(correct)
        true_positives = correct.sum()
        false_negatives = total - true_positives

        tpr = true_positives / total if total > 0 else 0
        fpr = 1 - tpr  # Simplified interpretation

        results.append({
            'Group': group,
            'Accuracy (TPR)': round(tpr, 3),
            'Error Rate (FPR)': round(fpr, 3),
            'Prediction Type': pred_col
        })

    return pd.DataFrame(results)

# ---- EO Gap ----
def calculate_gap(fairness_df):
    tpr_gap = fairness_df['Accuracy (TPR)'].max() - fairness_df['Accuracy (TPR)'].min()
    fpr_gap = fairness_df['Error Rate (FPR)'].max() - fairness_df['Error Rate (FPR)'].min()
    return round(tpr_gap + fpr_gap, 3)

# ---- Process ----
df_path = "/Users/alexlawson/Desktop/LLM-eval/merged_augmented_questions.csv"
df = pd.read_csv(df_path)
augmented_fairness = compute_fairness_from_accuracy(df, pred_col='Is Correct Augmented')


# ---- Save fairness metrics ----
augmented_fairness.to_csv("fairness_metrics_by_group.csv", index=False)

# ---- Plot ----
plt.figure(figsize=(10, 6))
sns.barplot(data=augmented_fairness, x='Group', y='Accuracy (TPR)', hue='Prediction Type')
plt.title("Model Accuracy (TPR) by Demographic Group")
plt.ylim(0, 0.4)
plt.ylabel("Accuracy (TPR)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("fairness_plot_tpr.png", dpi=300)
plt.close()

# ---- EO Gaps ----
eo_gap_augmented = calculate_gap(augmented_fairness)
print(f"EO Gap (Augmented): {eo_gap_augmented}")



df_path = "/Users/alexlawson/Desktop/LLM-eval/merged_augmented_questions.csv"
df = pd.read_csv(df_path)
#original_fairness = compute_fairness_from_accuracy(df, pred_col='Is Correct Original')
augmented_fairness = compute_fairness_from_accuracy(df, pred_col='Is Correct Augmented')


print("\nAugmented:")
print(augmented_fairness)
