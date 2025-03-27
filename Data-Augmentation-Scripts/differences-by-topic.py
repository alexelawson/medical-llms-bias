import os
import math
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# === File Paths ===
input_file = "/Users/alexlawson/Desktop/LLM-eval/merged_augmented_questions.csv"
output_dir = "/Users/alexlawson/Desktop/LLM-eval/Topic Breakdown"

# === Create output folder if it doesn't exist ===
os.makedirs(output_dir, exist_ok=True)

# === Load Data ===
df = pd.read_csv(input_file)

# --- Check for required columns ---
required_columns = ['topic_name', 'Is Correct Original', 'Is Correct Augmented', 'Demographic Variables']
for col in required_columns:
    if col not in df.columns:
        raise ValueError(f"Column '{col}' not found in the CSV. Please verify the column names.")

# === Compute Topic-Level Accuracy ===
# (Booleans are treated as 1 (True) and 0 (False); mean gives the proportion correct.)
original_accuracy = df.groupby('topic_name')['Is Correct Original'].mean().reset_index(name='Original % Correct')
augmented_accuracy = df.groupby('topic_name')['Is Correct Augmented'].mean().reset_index(name='Augmented % Correct')
topic_accuracy = pd.merge(original_accuracy, augmented_accuracy, on='topic_name')

# Round to 2 decimals for display
topic_accuracy['Original % Correct'] = topic_accuracy['Original % Correct'].round(2)
topic_accuracy['Augmented % Correct'] = topic_accuracy['Augmented % Correct'].round(2)

# Save topic-level accuracy table
topic_accuracy.to_csv(os.path.join(output_dir, "topic_accuracy.csv"), index=False)

# === Split Bar Plots by Topics ===
# Define how many topics per bar plot (adjust as needed)
topics_per_barplot = 15
topics_bar = topic_accuracy['topic_name'].tolist()
n_topics_bar = len(topics_bar)
num_bar_chunks = math.ceil(n_topics_bar / topics_per_barplot)

for i in range(num_bar_chunks):
    start = i * topics_per_barplot
    end = start + topics_per_barplot
    chunk_df = topic_accuracy.iloc[start:end]
    
    plt.figure(figsize=(12, 6))
    bar_width = 0.4
    x = range(len(chunk_df))
    plt.bar(x, chunk_df['Original % Correct'], width=bar_width, label='Original')
    plt.bar([j + bar_width for j in x], chunk_df['Augmented % Correct'], width=bar_width, label='Augmented')
    plt.xticks([j + bar_width/2 for j in x], chunk_df['topic_name'], rotation=45, ha='right')
    plt.ylabel('% Correct')
    plt.title(f'Original vs Augmented Accuracy by Topic (Chunk {i+1})')
    plt.legend()
    plt.tight_layout()
    barplot_chunk_file = os.path.join(output_dir, f"barplot_chunk_{i+1}.png")
    plt.savefig(barplot_chunk_file)
    plt.close()

# === Group by Topic and Demographic Variables ===
grouped = df.groupby(['topic_name', 'Demographic Variables']).agg(
    Original_Correct=('Is Correct Original', 'mean'),
    Augmented_Correct=('Is Correct Augmented', 'mean'),
    Count=('Is Correct Original', 'count')
).reset_index()

grouped['Original_Correct'] = grouped['Original_Correct'].round(2)
grouped['Augmented_Correct'] = grouped['Augmented_Correct'].round(2)

# Save grouped accuracy table
grouped.to_csv(os.path.join(output_dir, "topic_demographic_accuracy.csv"), index=False)

# === Prepare Data for Heatmaps (Using ALL Demographic Categories) ===
# Heatmap 1: Augmented Accuracy
pivot_augmented = grouped.pivot(index='topic_name', columns='Demographic Variables', values='Augmented_Correct')
pivot_augmented = pivot_augmented.fillna(0).astype(float)

# Heatmap 2: Accuracy Difference (Augmented - Original)
grouped['Accuracy_Diff'] = grouped['Augmented_Correct'] - grouped['Original_Correct']
pivot_diff = grouped.pivot(index='topic_name', columns='Demographic Variables', values='Accuracy_Diff')
pivot_diff = pivot_diff.fillna(0).astype(float)

# === Split Heatmaps by Topics ===
# Define how many topics per heatmap (adjust as needed)
topics_per_heatmap = 15
topics = list(pivot_augmented.index)
n_topics = len(topics)
num_chunks = math.ceil(n_topics / topics_per_heatmap)

# Loop over each chunk and create separate heatmap figures
for i in range(num_chunks):
    start = i * topics_per_heatmap
    end = start + topics_per_heatmap
    topics_chunk = topics[start:end]
    
    # Subset pivot tables for the chunk of topics
    pivot_aug_chunk = pivot_augmented.loc[topics_chunk]
    pivot_diff_chunk = pivot_diff.loc[topics_chunk]
    
    # --- Create a combined figure with two subplots for this chunk ---
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(18, 8), constrained_layout=True)
    
    # Heatmap of Augmented Accuracy for this chunk
    sns.heatmap(pivot_aug_chunk, annot=True, cmap='coolwarm', vmin=0, vmax=1, ax=ax1)
    ax1.set_title(f"Augmented Accuracy\nTopics {start+1} to {min(end, n_topics)}")
    ax1.set_ylabel("Topic")
    ax1.set_xlabel("Demographic Variable")
    
    # Heatmap of Accuracy Difference for this chunk
    sns.heatmap(pivot_diff_chunk, annot=True, cmap='bwr', center=0, ax=ax2)
    ax2.set_title(f"Accuracy Difference (Augmented - Original)\nTopics {start+1} to {min(end, n_topics)}")
    ax2.set_ylabel("Topic")
    ax2.set_xlabel("Demographic Variable")
    
    # Save the figure for this chunk
    heatmap_file = os.path.join(output_dir, f"heatmaps_chunk_{i+1}.png")
    plt.savefig(heatmap_file)
    plt.close()

print("✅ Analysis complete. CSV outputs, bar plot chunks, and heatmap chunks have been saved in the 'Topic Breakdown' folder.")
