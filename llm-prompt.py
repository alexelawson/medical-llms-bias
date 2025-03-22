#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
LLM Medical Question Evaluation Script

This script evaluates an LLM's performance on medical multiple choice questions,
comparing results between original and augmented questions.
"""

import os
import re
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch
import tqdm
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def setup_environment(cache_dir):
    """Set up environment variables for caching."""
    os.environ["HF_HOME"] = str(cache_dir)
    os.environ["TRANSFORMERS_CACHE"] = str(cache_dir)


def load_model(model_key, device_map="auto"):
    """
    Load the model and tokenizer.
    
    Args:
        model_key: HuggingFace model identifier
        device_map: Device mapping strategy
        
    Returns:
        tokenizer, model
    """
    print(f"Loading model: {model_key}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_key)
    
    model = AutoModelForCausalLM.from_pretrained(
        model_key,
        device_map=device_map,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True
    )
    
    return tokenizer, model


def load_dataset(file_path, single_choice_only=True):
    """
    Load the dataset from a parquet file.
    
    Args:
        file_path: Path to the parquet file
        single_choice_only: If True, filter only single choice questions
        
    Returns:
        DataFrame with loaded data
    """
    print(f"Loading dataset from {file_path}")
    df = pd.read_parquet(file_path)
    
    if single_choice_only:
        df = df[df["choice_type"] == "single"]
        print(f"Filtered to {len(df)} single choice questions")
    
    return df


def format_prompt(question, a, b, c, d):
    """Format a multiple choice question prompt."""
    return (
        f"Answer the following multiple choice question. Format your answer as a single number corresponding to the correct answer.\n"
        f"Question: {question}\n"
        f"1. {a}\n"
        f"2. {b}\n"
        f"3. {c}\n"
        f"4. {d}\n\n"
        f"Example Answer: 3. {c}\n"
        f"Your Answer: "
    )


def query_llm_local(tokenizer, model, prompt):
    """
    Query the local LLM model.
    
    Args:
        tokenizer: The tokenizer to use
        model: The model to query
        prompt: The prompt to send to the model
        
    Returns:
        The extracted answer (1-4) or None if no valid answer found
    """
    inputs = tokenizer(prompt, return_tensors="pt")
    
    # Move inputs to the same device as the model's first parameter
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=5,  # Forces a short response
            temperature=0.1,  # Ensures deterministic output
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )

    response_text = tokenizer.decode(
        output[:, inputs["input_ids"].shape[-1]:][0], 
        skip_special_tokens=True
    ).strip()

    # Extract the answer number
    pattern = r"([1-4])\."
    match = re.search(pattern, response_text)
    if match:
        return match.group(1)  # Extract the number
    else:
        print(f"No match found in response_text: {response_text}")
        return None  # No valid answer found


def evaluate_questions(df, tokenizer, model, num_samples=None, verbose=False):
    """
    Evaluate the model on both original and augmented questions.
    
    Args:
        df: DataFrame with questions
        tokenizer: The tokenizer to use
        model: The model to query
        num_samples: Number of samples to evaluate (None for all)
        verbose: Whether to print detailed output
        
    Returns:
        DataFrame with evaluation results
    """
    demographic_variables = ["Male", "Female", "White", "Black", "Asian", "Arab", "Other", "Low", "Middle", "High"]
    
    if num_samples:
        df = df.head(num_samples)
    
    results = []
    
    for _, row in tqdm.tqdm(df.iterrows(), total=len(df), desc="Evaluating questions"):
        original_question = row["question"]
        augmented_question = row["Augmented_Question"]
        opa = row["opa"]
        opb = row["opb"]
        opc = row["opc"]
        opd = row["opd"]
        correct_answer = str(int(row["cop"]) + 1)  # Convert to 1-indexed
        
        # Format the prompts
        prompt_original = format_prompt(original_question, opa, opb, opc, opd)
        prompt_augmented = format_prompt(augmented_question, opa, opb, opc, opd)
        
        # Query the model
        llm_answer_original = query_llm_local(tokenizer, model, prompt_original)
        llm_answer_augmented = query_llm_local(tokenizer, model, prompt_augmented)
        
        if verbose:
            print(f"Original Q: {original_question}")
            print(f"Augmented Q: {augmented_question}")
            print(f"Correct: {correct_answer}")
            print(f"LLM (Original): {llm_answer_original}")
            print(f"LLM (Augmented): {llm_answer_augmented}")
            print("-" * 50)
        
        # Check correctness
        is_correct_original = llm_answer_original and llm_answer_original == correct_answer
        is_correct_augmented = llm_answer_augmented and llm_answer_augmented == correct_answer
        
        # Identify the primary demographic
        selected_demographic = row[demographic_variables].idxmax()
        
        # Store results
        results.append({
            "Question": original_question,
            "Augmented Question": augmented_question,
            "Correct Answer": correct_answer,
            "LLM Answer Original": llm_answer_original,
            "Is Correct Original": is_correct_original,
            "LLM Answer Augmented": llm_answer_augmented,
            "Is Correct Augmented": is_correct_augmented,
            "Demographic Variables": selected_demographic
        })
    
    return pd.DataFrame(results)


def compute_summary_metrics(df):
    """
    Compute metrics from the evaluation results.
    
    Args:
        df: DataFrame with evaluation results
        
    Returns:
        overall_metrics: Dict with overall metrics
        group_metrics: DataFrame with per-demographic metrics
    """
    # Overall Accuracy
    overall_accuracy_original = df["Is Correct Original"].mean()
    overall_accuracy_augmented = df["Is Correct Augmented"].mean()
    
    # Helper function to compute odds: odds = p / (1 - p)
    def compute_odds(p):
        # Avoid division by zero: if p is 1, return infinity.
        return p / (1 - p) if p < 1 else float('inf')
    
    overall_odds_original = compute_odds(overall_accuracy_original)
    overall_odds_augmented = compute_odds(overall_accuracy_augmented)
    
    # Odds Ratio: odds (original) divided by odds (augmented)
    overall_odds_ratio = overall_odds_original / overall_odds_augmented if overall_odds_augmented != 0 else float('inf')
    
    # Accuracy Parity: Difference between augmented and original accuracy
    overall_accuracy_parity = overall_accuracy_augmented - overall_accuracy_original
    
    overall_metrics = {
        "overall_accuracy_original": overall_accuracy_original,
        "overall_accuracy_augmented": overall_accuracy_augmented,
        "overall_odds_original": overall_odds_original,
        "overall_odds_augmented": overall_odds_augmented,
        "overall_odds_ratio": overall_odds_ratio,
        "overall_accuracy_parity": overall_accuracy_parity
    }
    
    # Per-demographic metrics
    group_metrics = df.groupby("Demographic Variables").agg(
        accuracy_original=("Is Correct Original", "mean"),
        accuracy_augmented=("Is Correct Augmented", "mean")
    ).reset_index()
    
    # Compute odds for each group
    group_metrics["odds_original"] = group_metrics["accuracy_original"].apply(compute_odds)
    group_metrics["odds_augmented"] = group_metrics["accuracy_augmented"].apply(compute_odds)
    
    # Compute odds ratio per group (handling division by zero)
    group_metrics["odds_ratio"] = group_metrics.apply(
        lambda row: row["odds_original"] / row["odds_augmented"] if row["odds_augmented"] != 0 else float('inf'),
        axis=1
    )
    
    # Accuracy parity per group: augmented accuracy minus original accuracy
    group_metrics["accuracy_parity"] = group_metrics["accuracy_augmented"] - group_metrics["accuracy_original"]
    
    return overall_metrics, group_metrics


def plot_metrics(overall_metrics, group_metrics, output_dir):
    """
    Create and save plots of the metrics.
    
    Args:
        overall_metrics: Dict with overall metrics
        group_metrics: DataFrame with per-demographic metrics
        output_dir: Directory to save plots
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot overall accuracy
    labels = ['Original', 'Augmented']
    accuracies = [
        overall_metrics["overall_accuracy_original"],
        overall_metrics["overall_accuracy_augmented"]
    ]
    
    plt.figure(figsize=(6, 4))
    bars = plt.bar(labels, accuracies, color=['blue', 'orange'])
    plt.ylim(0, 1)
    plt.ylabel("Accuracy")
    plt.title("Overall Accuracy Comparison")
    
    # Annotate each bar with its value
    for bar in bars:
        height = bar.get_height()
        plt.annotate(f"{height:.2f}", 
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3), textcoords="offset points",
                ha='center', va='bottom')
    
    plt.savefig(os.path.join(output_dir, "overall_accuracy.png"))
    plt.close()
    
    # Plot group accuracy
    groups = group_metrics["Demographic Variables"]
    accuracy_orig = group_metrics["accuracy_original"]
    accuracy_aug = group_metrics["accuracy_augmented"]
    
    x = np.arange(len(groups))
    width = 0.35
    
    plt.figure(figsize=(10, 6))
    plt.bar(x - width/2, accuracy_orig, width, label="Original", color='blue')
    plt.bar(x + width/2, accuracy_aug, width, label="Augmented", color='orange')
    
    plt.ylabel("Accuracy")
    plt.title("Accuracy by Demographic Group")
    plt.xticks(x, groups, rotation=45)
    plt.legend()
    plt.tight_layout()
    
    plt.savefig(os.path.join(output_dir, "group_accuracy.png"))
    plt.close()
    
    # Plot accuracy parity
    plt.figure(figsize=(10, 6))
    bars = plt.bar(groups, group_metrics["accuracy_parity"], color='purple')
    plt.ylabel("Accuracy Parity")
    plt.title("Accuracy Parity by Demographic Group (Augmented - Original)")
    
    # Annotate each bar with its value
    for bar in bars:
        height = bar.get_height()
        plt.annotate(f"{height:.2f}",
                xy=(bar.get_x() + bar.get_width()/2, height),
                xytext=(0, 3), textcoords="offset points",
                ha='center', va='bottom')
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "group_accuracy_parity.png"))
    plt.close()


def save_results(results_df, overall_metrics, group_metrics, output_dir):
    """
    Save results to CSV files.
    
    Args:
        results_df: DataFrame with evaluation results
        overall_metrics: Dict with overall metrics
        group_metrics: DataFrame with per-demographic metrics
        output_dir: Directory to save results
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Save results
    results_df.to_csv(os.path.join(output_dir, "results.csv"), index=False)
    
    # Save overall metrics
    overall_metrics_df = pd.DataFrame([overall_metrics])
    overall_metrics_df.to_csv(os.path.join(output_dir, "overall_metrics.csv"), index=False)
    
    # Save group metrics
    group_metrics.to_csv(os.path.join(output_dir, "group_metrics.csv"), index=False)
    
    print(f"Results saved to {output_dir}")


def main():
    """Main function to run the script."""
    parser = argparse.ArgumentParser(description="Evaluate LLM performance on medical multiple choice questions")
    
    # Model options
    parser.add_argument("--model", type=str, default="BioMistral/BioMistral-7B", # default="meta-llama/Llama-3.2-1B-Instruct",
                      help="HuggingFace model to use")
    # Data options
    parser.add_argument("--data", type=str, default="augmented_dataset.parquet",
                      help="Path to the parquet file with questions (default: augmented_dataset.parquet)")
    parser.add_argument("--samples", type=int, default=None,
                      help="Number of samples to evaluate (default: all)")
    
    # Output options
    parser.add_argument("--output-dir", type=str, default="./results",
                      help="Directory to save results and plots")
    parser.add_argument("--cache-dir", type=str, default="./cache",
                      help="Directory for model cache")
    
    # Other options
    parser.add_argument("--verbose", action="store_true",
                      help="Print detailed output")
    
    args = parser.parse_args()
    
    # Set up environment
    workdir = Path.cwd()
    cache_dir = Path(args.cache_dir)
    setup_environment(cache_dir)
    
    # Load model and tokenizer
    tokenizer, model = load_model(args.model)
    
    # Load dataset
    df = load_dataset(args.data)

    df = df.head(60)
    
    # Evaluate questions
    results_df = evaluate_questions(df, tokenizer, model, args.samples, args.verbose)
    
    # Compute metrics
    overall_metrics, group_metrics = compute_summary_metrics(results_df)
    
    # Plot and save results
    plot_metrics(overall_metrics, group_metrics, args.output_dir)
    save_results(results_df, overall_metrics, group_metrics, args.output_dir)
    
    print("Evaluation completed.")


if __name__ == "__main__":
    main()