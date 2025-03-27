import pandas as pd


df_changes = pd.read_csv("/Users/alexlawson/Desktop/LLM-eval/results(in).csv")
df_details = pd.read_csv("/Users/alexlawson/Desktop/LLM-eval/augmented_dataset.csv")

merged_df = pd.merge(
    df_changes,            # e.g., contains answer change info
    df_details,            # e.g., contains metadata/details
    left_on="Augmented Question",
    right_on="Augmented_Question",
    how="inner"
)

merged_df.to_csv("/Users/alexlawson/Desktop/LLM-eval/merged_augmented_questions.csv", index=False)

