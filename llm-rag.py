#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
LLM Medical Question Evaluation Script

This script evaluates an LLM's performance on medical multiple choice questions,
comparing results between original and augmented questions.
"""

import os
import sys
import re
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch
import tqdm
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from src.medrag import MedRAG

"""
move to MedRAG directory and run:
python llm-rag.py --data ../Datasets/GPT3.5turbo-augmentedquestions02.csv --output-dir ../results_rag/ --cache-dir ../.cache --start 0 --end 1557
"""


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
    file_path = file_path.rstrip('/')
    if file_path.endswith('.parquet'):
        df = pd.read_parquet(file_path)
    elif file_path.endswith('.csv'):
        df = pd.read_csv(file_path)
    else:
        raise ValueError(f"Unsupported file format: {file_path}")
    
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


def query_llm_local(medrag, question, a, b, c, d, k=32):
    """
    Query using MedRAG for retrieval-augmented medical question answering.
    
    Args:
        medrag: Pre-initialized MedRAG instance
        question: The medical question to answer
        a, b, c, d: The answer options
        k: Number of snippets to retrieve
        
    Returns:
        The extracted answer (1-4) or None if no valid answer found
    """
    # Format options dictionary as required by MedRAG
    options = {
        "A": a,
        "B": b,
        "C": c,
        "D": d
    }
    
    # Get answer, snippets and scores
    answer, snippets, scores = medrag.answer(question=question, options=options, k=k)
    
    # Extract the numeric answer (1-4) from the result
    letter_to_number = {"A": "1", "B": "2", "C": "3", "D": "4"}
    
    # Extract the letter from the answer
    answer_letter = None
    for letter in letter_to_number.keys():
        if letter in answer:
            answer_letter = letter
            break
    
    if answer_letter:
        return letter_to_number[answer_letter]
    else:
        print(f"No valid answer letter found in response: {answer}")
        return None


def evaluate_questions(df, medrag, num_samples=None, verbose=False, k=32):
    """
    Evaluate the model on both original and augmented questions using RAG.
    
    Args:
        df: DataFrame with questions
        medrag: Pre-initialized MedRAG instance
        num_samples: Number of samples to evaluate (None for all)
        verbose: Whether to print detailed output
        k: Number of snippets to retrieve
        
    Returns:
        DataFrame with evaluation results
    """
    if num_samples:
        df = df.head(num_samples)
    
    results = []
    
    for _, row in tqdm.tqdm(df.iterrows(), total=len(df), desc="Evaluating questions"):
        original_question = row["question"]
        augmented_question = row["Augmented Question"]
        opa = row["opa"]
        opb = row["opb"]
        opc = row["opc"]
        opd = row["opd"]
        correct_answer = str(int(row["cop"]) + 1)  # Convert to 1-indexed
        
        # Query using RAG instead of direct LLM
        llm_answer_original = query_llm_local(medrag, original_question, opa, opb, opc, opd, k)
        llm_answer_augmented = query_llm_local(medrag, augmented_question, opa, opb, opc, opd, k)
        
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
        
        # Get demographic directly from the Demographic column
        demographic = row["Demographic"]
        
        # Store results
        results.append({
            "Question": original_question,
            "Augmented Question": augmented_question,
            "Correct Answer": correct_answer,
            "LLM Answer Original": llm_answer_original,
            "Is Correct Original": is_correct_original,
            "LLM Answer Augmented": llm_answer_augmented,
            "Is Correct Augmented": is_correct_augmented,
            "Demographic Variables": demographic  # Keep the same key name for compatibility with downstream functions
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
    parser.add_argument("--model", type=str, default="axiong/PMC_LLaMA_13B_int8",
                      help="LLM to use for MedRAG")
    # Data options
    parser.add_argument("--data", type=str, default="augmented_dataset.parquet",
                      help="Path to the parquet file with questions (default: augmented_dataset.parquet)")
    parser.add_argument("--samples", type=int, default=60,
                      help="Number of samples to evaluate (default: all)")
    
    # RAG options
    parser.add_argument("--rag", action="store_true", default=True,
                      help="Use Retrieval-Augmented Generation")
    parser.add_argument("--retriever", type=str, default="MedCPT",
                      help="Retriever to use (default: MedCPT)")
    parser.add_argument("--corpus", type=str, default="PubMed",
                      help="Corpus to use (default: PubMed)")
    parser.add_argument("--k", type=int, default=32,
                      help="Number of snippets to retrieve (default: 32)")
    
    # Output options
    parser.add_argument("--output-dir", type=str, default="./results",
                      help="Directory to save results and plots")
    parser.add_argument("--cache-dir", type=str, default="./cache",
                      help="Directory for model cache")
    
    # Other options
    parser.add_argument("--verbose", action="store_true",
                      help="Print detailed output")
    
    # Modify start and end arguments to be required
    parser.add_argument("--start", type=int, required=True,
                      help="Starting index for data chunking")
    parser.add_argument("--end", type=int, required=True,
                      help="Ending index for data chunking")
    
    args = parser.parse_args()
    
    # Modify output directory to include start and end indices
    base_output_dir = args.output_dir.rstrip('/')  # Remove trailing slash if present
    args.output_dir = f"{base_output_dir}_{args.start}_{args.end}"
    
    # Set up environment
    workdir = Path.cwd()
    cache_dir = Path(args.cache_dir)
    setup_environment(cache_dir)
    
    # Load dataset
    df = load_dataset(args.data)
    
    # Validate and apply chunking
    if args.start >= len(df):
        args.start = len(df) - 1
    if args.end > len(df):
        args.end = len(df)
    if args.start >= args.end:
        raise ValueError(f"Start index ({args.start}) must be less than end index ({args.end})")
    
    df = df.iloc[args.start:args.end]
    
    # Initialize MedRAG once
    print("Initializing MedRAG...")
    medrag = MedRAG(
        llm_name=args.model,
        rag=args.rag,
        retriever_name=args.retriever,
        corpus_name=args.corpus
    )
    
    # Evaluate questions with RAG
    results_df = evaluate_questions(
        df,
        medrag,
        args.samples, 
        args.verbose,
        k=args.k
    )
    
    # Compute metrics
    overall_metrics, group_metrics = compute_summary_metrics(results_df)
    
    # Plot and save results
    plot_metrics(overall_metrics, group_metrics, args.output_dir)
    save_results(results_df, overall_metrics, group_metrics, args.output_dir)
    
    print("Evaluation completed.")


if __name__ == "__main__":
    main()
