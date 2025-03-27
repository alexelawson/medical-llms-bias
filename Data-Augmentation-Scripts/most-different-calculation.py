import os
import math
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# === File Paths ===
input_file = "/Users/alexlawson/Desktop/LLM-eval/merged_augmented_questions.csv"
output_dir = "/Users/alexlawson/Desktop/LLM-eval/Topic Breakdown/BadSubjects"

# === Create output folder if it doesn't exist ===
os.makedirs(output_dir, exist_ok=True)

# === Load Data ===
df = pd.read_csv(input_file)

# --- Check for required columns ---
required_columns = ['subject_name', 'Is Correct Original', 'Is Correct Augmented', 'Demographic Variables']
for col in required_columns:
    if col not in df.columns:
        raise ValueError(f"Column '{col}' not found in the CSV. Please verify the column names.")

# === Compute Overall Topic-Level Accuracy ===
# (Since booleans are treated as 1 (True) and 0 (False), .mean() gives the proportion correct.)
original_accuracy = df.groupby('subject_name')['Is Correct Original'].mean().reset_index(name='Original % Correct')
augmented_accuracy = df.groupby('subject_name')['Is Correct Augmented'].mean().reset_index(name='Augmented % Correct')
topic_accuracy = pd.merge(original_accuracy, augmented_accuracy, on='subject_name')

# Round values to 2 decimals
topic_accuracy['Original % Correct'] = topic_accuracy['Original % Correct'].round(2)
topic_accuracy['Augmented % Correct'] = topic_accuracy['Augmented % Correct'].round(2)

# Compute the overall difference (Augmented - Original)
topic_accuracy['Diff'] = topic_accuracy['Augmented % Correct'] - topic_accuracy['Original % Correct']

# Save overall topic accuracy for reference
topic_accuracy.to_csv(os.path.join(output_dir, "subject_accuracy.csv"), index=False)

# === Filter for Topics with Significantly Worse Augmented Performance ===
# Define a threshold: default is -0.2 (i.e., a drop of 20 percentage points or more)
threshold = -0.05
signif_topics = topic_accuracy[topic_accuracy['Diff'] < threshold]['subject_name'].tolist()

# Save these topics to a CSV for reference
topic_accuracy[topic_accuracy['Diff'] < threshold].to_csv(os.path.join(output_dir, "significantly_worse_topics.csv"), index=False)

# === Group by Topic and Demographic Variables ===
grouped = df.groupby(['subject_name', 'Demographic Variables']).agg(
    Original_Correct=('Is Correct Original', 'mean'),
    Augmented_Correct=('Is Correct Augmented', 'mean'),
    Count=('Is Correct Original', 'count')
).reset_index()

# Round the grouped values for display
grouped['Original_Correct'] = grouped['Original_Correct'].round(2)
grouped['Augmented_Correct'] = grouped['Augmented_Correct'].round(2)

# Save the grouped demographic breakdown
grouped.to_csv(os.path.join(output_dir, "subject_demographic_accuracy.csv"), index=False)

# === Filter the Grouped Data to Only Include the Significantly Worse Topics ===
grouped_signif = grouped[grouped['subject_name'].isin(signif_topics)].copy()

# Compute Accuracy Difference for each demographic group (if not already computed)
grouped_signif['Accuracy_Diff'] = grouped_signif['Augmented_Correct'] - grouped_signif['Original_Correct']

# === Prepare a Pivot Table for the Heatmap (Accuracy Difference) ===
pivot_diff_signif = grouped_signif.pivot(index='subject_name', columns='Demographic Variables', values='Accuracy_Diff')
pivot_diff_signif = pivot_diff_signif.fillna(0).astype(float)

# === Split the Heatmap into Chunks if Necessary ===
# Define how many topics per heatmap (adjust as needed)
topics_per_heatmap_signif = 15
topics_signif = list(pivot_diff_signif.index)
n_signif = len(topics_signif)
num_chunks_signif = math.ceil(n_signif / topics_per_heatmap_signif)

for j in range(num_chunks_signif):
    start = j * topics_per_heatmap_signif
    end = start + topics_per_heatmap_signif
    topics_chunk = topics_signif[start:end]
    
    # Subset the pivot table for these topics
    pivot_chunk = pivot_diff_signif.loc[topics_chunk]
    
    plt.figure(figsize=(12, 8))
    sns.heatmap(pivot_chunk, annot=True, cmap='bwr', center=0)
    plt.title(f"Accuracy Difference (Augmented - Original)\nSignificantly Worse Topics {start+1} to {min(end, n_signif)}\n(Threshold: {threshold})")
    plt.ylabel("Subject")
    plt.xlabel("Demographic Variable")
    plt.tight_layout()
    
    # Save this heatmap chunk
    signif_heatmap_file = os.path.join(output_dir, f"significantly_worse_heatmap_chunk_{j+1}.png")
    plt.savefig(signif_heatmap_file)
    plt.close()

print("✅ Analysis complete. Heatmaps for topics with significantly worse augmented performance have been saved in the 'Topic Breakdown' folder.")
