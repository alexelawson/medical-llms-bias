import pandas as pd
import matplotlib.pyplot as plt

# Load your dataset into a DataFrame
df = pd.read_csv("/Users/alexlawson/Desktop/LLM-eval/merged_augmented_questions.csv")

# ------------------------------
# 1. Overall unique counts
# ------------------------------

# Count of unique questions overall
unique_questions = df['question'].nunique()

# Count of unique augmented questions overall
unique_augmented_questions = df['Augmented_Question'].nunique()

# Create an overall summary DataFrame
overall_stats = pd.DataFrame({
    'Metric': ['Unique Questions', 'Unique Augmented Questions'],
    'Count': [unique_questions, unique_augmented_questions]
})

# ------------------------------
# 2. Unique questions per subject
# ------------------------------

# Group by 'subject_name' and count unique questions
unique_questions_by_subject = (
    df.groupby('subject_name')['question']
      .nunique()
      .reset_index()
      .rename(columns={'question': 'unique_questions'})
)

# ------------------------------
# 3. Unique augmented questions per demographic
# ------------------------------

# Group by 'Demographic Variables' and count unique augmented questions
unique_augmented_by_demographic = (
    df.groupby('Demographic Variables')['Augmented_Question']
      .nunique()
      .reset_index()
      .rename(columns={'Augmented_Question': 'unique_augmented_questions'})
)

# ------------------------------
# Print DataFrames for verification
# ------------------------------
print("Overall Stats:")
print(overall_stats)
print("\nUnique Questions by Subject:")
print(unique_questions_by_subject)
print("\nUnique Augmented Questions by Demographic:")
print(unique_augmented_by_demographic)

# ------------------------------
# Visualizations
# ------------------------------

# Bar chart for Unique Questions by Subject
plt.figure(figsize=(10, 6))
plt.bar(unique_questions_by_subject['subject_name'],
        unique_questions_by_subject['unique_questions'],
        color='skyblue')
plt.xlabel('Subject Name')
plt.ylabel('Unique Questions')
plt.title('Unique Questions by Subject')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("unique_questions_by_subject.png")
plt.show()

# Bar chart for Unique Augmented Questions by Demographic
plt.figure(figsize=(10, 6))
plt.bar(unique_augmented_by_demographic['Demographic Variables'],
        unique_augmented_by_demographic['unique_augmented_questions'],
        color='salmon')
plt.xlabel('Demographic Variables')
plt.ylabel('Unique Augmented Questions')
plt.title('Unique Augmented Questions by Demographic')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("unique_augmented_by_demographic.png")
plt.show()
