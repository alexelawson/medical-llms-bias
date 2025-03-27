import pandas as pd

# Load the dataset
df = pd.read_csv('/Users/alexlawson/Desktop/LLM-eval/augmented_dataset.csv')

# Filter rows where subject_name is 'Psychiatry' and choice_type is 'Single'
filtered_df = df[(df['subject_name'] == 'Psychiatry') & (df['choice_type'] == 'single')]

# Remove the specified columns
columns_to_drop = [
    'Augmented_Question', 'Gender', 'Race', 'SES',
    'Male', 'Female', 'White', 'Black', 'Arab',
    'Asian', 'Other', 'Low', 'Middle', 'High'
]
filtered_df = filtered_df.drop(columns=columns_to_drop, errors='ignore')

# Remove duplicate rows based on the 'question' column so that each question appears only once
filtered_df = filtered_df.drop_duplicates(subset='question')

# Randomly sample 200 questions (using random_state for reproducibility)
sample_df = filtered_df.sample(n=200, random_state=42)

# Save the sampled DataFrame to a new CSV file
sample_df.to_csv('/Users/alexlawson/Desktop/LLM-eval/sample_200_questions.csv', index=False)

# Save the filtered DataFrame to a new CSV file
#filtered_df.to_csv('/Users/alexlawson/Desktop/LLM-eval/filtered_dataset-short.csv', index=False)
