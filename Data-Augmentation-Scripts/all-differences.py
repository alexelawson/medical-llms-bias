import os
import math
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# === File Paths ===
input_file = "/Users/alexlawson/Desktop/LLM-eval/merged_augmented_questions.csv"
output_dir = "/Users/alexlawson/Desktop/LLM-eval/Topic Breakdown/AllTopics"

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

# === Compute Accuracy Difference for each demographic group ===
grouped['Accuracy_Diff'] = grouped['Augmented_Correct'] - grouped['Original_Correct']

# === Prepare a Pivot Table for the Heatmap (Accuracy Difference) ===
pivot_diff_all = grouped.pivot(index='subject_name', columns='Demographic Variables', values='Accuracy_Diff')
pivot_diff_all = pivot_diff_all.fillna(0).astype(float)

# === Split the Heatmap into Chunks if Necessary ===
#topics_per_heatmap = 50
#topics_all = list(pivot_diff_all.index)
#n_all = len(topics_all)
#num_chunks = math.ceil(n_all / topics_per_heatmap)

#for j in range(num_chunks):
    #start = j * topics_per_heatmap
    #end = start + topics_per_heatmap
    #topics_chunk = topics_all[start:end]
    
    # Subset the pivot table for these topics
   # pivot_chunk = pivot_diff_all.loc[topics_chunk]
    
plt.figure(figsize=(12, 8))
sns.heatmap(pivot_diff_all, annot=True, cmap='bwr', center=0)
plt.title(f"Accuracy Difference (Augmented - Original)")
plt.ylabel("Subject")
plt.xlabel("Demographic Variable")
plt.tight_layout()
    
    # Save this heatmap chunk
heatmap_file = os.path.join(output_dir, f"all_topics_heatmap_chunk.png")
plt.savefig(heatmap_file)
plt.close()

print("✅ Analysis complete. Heatmaps for all topics have been saved in the 'Topic Breakdown/AllTopics' folder.")
