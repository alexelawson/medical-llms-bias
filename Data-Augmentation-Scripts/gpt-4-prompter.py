import pandas as pd
import openai
import time

# Set your OpenAI API key
openai.api_key = "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"  # replace with your API key

def augment_question(question, demographic):
    """
    Uses ChatGPT (gpt-3.5-turbo) to generate an augmented version of the question
    incorporating the given demographic variable.
    """
    prompt = (
        f"Please modify the following prompt by adding or changing it to include the demographic variable."
        f"'{demographic}'.\n\n"
        f"Prompt: {question}\n\n"
        f"Augmented Prompt:"
    )
    
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=100,
            n=1,
            temperature=0.5,
        )
        augmented_question = response.choices[0].message['content'].strip()
    except Exception as e:
        print(f"Error generating augmentation for demographic '{demographic}': {e}")
        augmented_question = None
    return augmented_question

# Load the dataset (assuming you want to use the 200-question sample generated previously)
df = pd.read_csv('/Users/alexlawson/Desktop/LLM-eval/sample_200_questions.csv')

# Define the demographic variables for augmentation
demographics = [
    "Male", "Female", "Low Socioeconomic Status", "High Socioeconomic Status",
    "Black", "White", "Arab", "Asian", "Other"
]

augmented_rows = []

def test_augmentation(prompt, demographic):
    prompt = (
        f"Please modify the following prompt by adding or changing it to include the demographic variable '{demographic}'.\n\n"
        f"Prompt: {prompt}\n\n"
        f"Augmented Prompt:"
    )
    
    response = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are an assistant that augments prompts by integrating demographic variables. You should be concise."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=50,
        n=1,
        temperature=0.1,
    )
    augmented_question = response.choices[0].message.content.strip()
    return augmented_question

for idx, row in df.iterrows():
    question = row['question']
    for demo in demographics:
        print(demo)
        print(question)
        augmented_q = test_augmentation(question, demo)
        # Create a copy of the row with the augmented question and demographic info
        new_row = row.copy()
        new_row["Augmented Question"] = augmented_q
        new_row["Demographic"] = demo  # Optional: helps track which augmentation corresponds to which demographic
        augmented_rows.append(new_row)
        # Optional: delay to help avoid rate limiting
        time.sleep(1)

# Create a new DataFrame from the augmented rows
augmented_df = pd.DataFrame(augmented_rows)

# Save the new DataFrame to a CSV file
augmented_df.to_csv('/Users/alexlawson/Desktop/LLM-eval/augmented_questions.csv', index=False)

print("Augmented questions saved successfully.")
