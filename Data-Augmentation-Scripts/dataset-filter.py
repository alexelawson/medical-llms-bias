import pandas as pd

# Load the dataset
df = pd.read_csv('/Users/alexlawson/Desktop/LLM-eval/augmented_dataset.csv')

filtered_df = df[
    (
        (df['subject_name'] == 'Pharmacology') | (df['subject_name'] == 'Psychiatry')
    ) &
    (df['choice_type'].str.lower() == 'single') &
    (df['question'].str.contains('patient', case=False, na=False)) &
    (~df['question'].str.contains('male|female', case=False, na=False)) &
    (~df['question'].str.contains('man|woman', case=False, na=False)) &
    (~df['question'].str.contains('boy|girl', case=False, na=False)) &
    (~df['question'].str.contains('he|she', case=False, na=False)) &
    (~df['question'].str.contains('his|her', case=False, na=False)) &
    (~df['question'].str.contains('pregnant|semen', case=False, na=False))
]

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
#sample_df = filtered_df.sample(n=200, random_state=42)

# Save the sampled DataFrame to a new CSV file
filtered_df.to_csv('/Users/alexlawson/Desktop/LLM-eval/patient_questions-filter-200.csv', index=False)

# Save the filtered DataFrame to a new CSV file
#filtered_df.to_csv('/Users/alexlawson/Desktop/LLM-eval/filtered_dataset-short.csv', index=False)
