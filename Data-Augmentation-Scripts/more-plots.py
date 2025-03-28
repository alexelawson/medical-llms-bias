import os
import math
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# === File Paths ===
input_file = "/Users/alexlawson/Desktop/LLM-eval/merged_augmented_questions.csv"
output_dir = "/Users/alexlawson/Desktop/LLM-eval/Topic Breakdown/moreplots"

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

plt.figure(figsize=(12, 8))
sns.scatterplot(
    data=grouped, 
    x='Original_Correct', 
    y='Augmented_Correct',
    hue='subject_name', 
    size='Count', 
    sizes=(20, 80),  # smaller bubbles
    alpha=0.6, 
    edgecolor='w',
    palette='tab10'
)
plt.plot([0.25, 0.6], [0.2, 0.6], 'r--')  # reference line for equality
plt.xlabel("Original Accuracy")
plt.ylabel("Augmented Accuracy")
plt.title("Scatter Plot: Original vs. Augmented Accuracy\n(Bubble size ~ Count, Colored by Subject)")

# Place the legend on the side
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.tight_layout()
plt.savefig("bubble_plot.png", bbox_inches="tight")
