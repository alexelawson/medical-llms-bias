import os
import math
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# === File Paths ===
input_file = "/Users/alexlawson/Desktop/LLM-eval/merged_augmented_questions.csv"
output_dir = "/Users/alexlawson/Desktop/LLM-eval/Topic Breakdown/AllTopics"

# Create output folder if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# === Load Data ===
df = pd.read_csv(input_file)

# --- Check for required columns ---
required_columns = ['subject_name', 'Is Correct Original', 'Is Correct Augmented', 'Demographic Variables']
for col in required_columns:
    if col not in df.columns:
        raise ValueError(f"Column '{col}' not found in the CSV. Please verify the column names.")

# === Compute Overall Subject-Level Accuracy (Original and Augmented) ===
original_accuracy = df.groupby('Demographic Variables')['Is Correct Original'].mean().reset_index(name='Original % Correct')
augmented_accuracy = df.groupby('Demographic Variables')['Is Correct Augmented'].mean().reset_index(name='Augmented % Correct')
topic_accuracy = pd.merge(original_accuracy, augmented_accuracy, on='Demographic Variables')

# Round values to 2 decimals
topic_accuracy['Original % Correct'] = topic_accuracy['Original % Correct'].round(2)
topic_accuracy['Augmented % Correct'] = topic_accuracy['Augmented % Correct'].round(2)
# Compute the difference (Augmented - Original)
topic_accuracy['Diff'] = topic_accuracy['Augmented % Correct'] - topic_accuracy['Original % Correct']

# Save overall topic accuracy for reference
topic_accuracy.to_csv(os.path.join(output_dir, "demo_accuracy.csv"), index=False)

# === Group by Subject and Demographic Variables ===
grouped = df.groupby(['subject_name', 'Demographic Variables']).agg(
    Original_Correct=('Is Correct Original', 'mean'),
    Augmented_Correct=('Is Correct Augmented', 'mean'),
    Count=('Is Correct Original', 'count')
).reset_index()

# Round the grouped values for display
grouped['Original_Correct'] = grouped['Original_Correct'].round(2)
grouped['Augmented_Correct'] = grouped['Augmented_Correct'].round(2)

# Compute the accuracy difference (Augmented - Original) for each demographic group
grouped['Accuracy_Diff'] = grouped['Augmented_Correct'] - grouped['Original_Correct']

# === Calculate Accuracy Parity per Subject ===
# Merge overall subject original accuracy into the grouped data
grouped = pd.merge(grouped, topic_accuracy[['Demographic Variables', 'Original % Correct']], on='Demographic Variables')
# Accuracy Parity: difference between the group’s original accuracy and the subject’s overall original accuracy
grouped['Accuracy_Parity'] = grouped['Original_Correct'] - grouped['Original % Correct']

# Save the grouped demographic breakdown with parity information
grouped.to_csv(os.path.join(output_dir, "demographic_accuracy_with_parity.csv"), index=False)

# === Compute Overall Demographic-Level Accuracy ===
# Here we aggregate across all subjects to see the overall accuracy for each demographic variable.
demographic_accuracy = df.groupby('Demographic Variables')['Is Correct Original'].mean().reset_index(name='Overall Demographic Accuracy')
demographic_accuracy['Overall Demographic Accuracy'] = demographic_accuracy['Overall Demographic Accuracy'].round(2)

# Calculate the overall dataset accuracy (across all subjects and demographics)
overall_accuracy = round(df['Is Correct Original'].mean(), 2)
# Accuracy Parity for demographics

def plot_original_vs_augmented(grouped_df, output_file, figsize=(12, 8)):
    """
    Creates and saves a grouped bar graph comparing 'Original Correct' and 'Augmented Correct'
    for each demographic variable.
    
    The plot displays:
      - Original Correct as a white bar with a black outline.
      - Augmented Correct as a white bar with a black outline and black stripes.
    
    Parameters:
        grouped_df (pd.DataFrame): DataFrame that must contain the columns:
            'Demographic Variables', 'Original_Correct', and 'Augmented_Correct'.
            If there are multiple rows per demographic, the function aggregates them by taking the mean.
        output_file (str): File path to save the plot image.
        figsize (tuple): Figure size for the plot (default: (12, 8)).
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    # If the DataFrame contains multiple rows per demographic, aggregate by mean.
    if grouped_df['Demographic Variables'].duplicated().any():
        df_agg = grouped_df.groupby('Demographic Variables', as_index=False).agg({
            'Original_Correct': 'mean',
            'Augmented_Correct': 'mean'
        })
    else:
        df_agg = grouped_df.copy()

    # Create positions for each demographic group on the x-axis.
    x = np.arange(len(df_agg))
    bar_width = 0.35  # width for each bar

    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot "Original Correct": white fill, black outline.
    bars1 = ax.bar(x - bar_width/2, df_agg['Original % Correct'], bar_width,
                   label='Original % Correct', color='white', edgecolor='black')
    
    # Plot "Augmented Correct": white fill, black outline, with hatch (black stripes).
    bars2 = ax.bar(x + bar_width/2, df_agg['Augmented % Correct'], bar_width,
                   label='Augmented % Correct', color='white', edgecolor='black', hatch='///')
    
    # Set the x-axis tick labels to demographic variable names.
    ax.set_xticks(x)
    ax.set_xticklabels(df_agg['Demographic Variables'], rotation=45, ha='right')
    
    # Add labels and title.
    ax.set_xlabel('Demographic Variables')
    ax.set_ylabel('Accuracy')
    ax.set_title('Original vs Augmented Correct by Demographic')
    ax.legend()
    ax.set_ylim(0.2, 0.4)
    ax.legend(loc='lower right')

    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()
    
    print(f"✅ Bar graph saved to {output_file}")



plot_original_vs_augmented(topic_accuracy, os.path.join(output_dir, "accuracy_parity-graph.png"))