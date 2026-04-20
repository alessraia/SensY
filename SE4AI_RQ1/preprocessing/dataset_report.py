import os
import json
import pandas as pd

def analyze_dataset(dataset_path, dataset_name):
    """
    Analyzes a dataset to count duplicates, questions with label 1, and questions with label 0.

    Args:
        dataset_path (str): Path to the JSON dataset.
        dataset_name (str): Name of the dataset for identification in the report.

    Returns:
        dict: A dictionary containing the analysis results.
    """
    print(f"Analyzing {dataset_name} dataset...")

    # Load the dataset
    with open(dataset_path, "r") as f:
        data = json.load(f)

    # Convert to DataFrame
    df = pd.DataFrame(data)

    # Ensure required columns exist
    if "question_en" not in df.columns or "sensitive?" not in df.columns:
        raise ValueError(f"{dataset_name} dataset must contain 'question' and 'sensitive?' columns.")

    # Analysis
    total_questions = len(df)
    duplicate_count = df["question_en"].duplicated().sum()
    sensitive_count = df[df["sensitive?"] == 1].shape[0]
    non_sensitive_count = df[df["sensitive?"] == 0].shape[0]

    print(f"Total questions: {total_questions}")
    print(f"Duplicate questions: {duplicate_count}")
    print(f"Sensitive questions (1): {sensitive_count}")
    print(f"Non-sensitive questions (0): {non_sensitive_count}")

    # Return results as a dictionary
    return {
        "dataset_name": dataset_name,
        "total_questions": total_questions,
        "duplicate_questions": duplicate_count,
        "sensitive_questions": sensitive_count,
        "non_sensitive_questions": non_sensitive_count,
    }

if __name__ == "__main__":
    # File paths
    train_path = "data/question_train.json"
    test_path = "data/question_test.json"
    augmented_train_path = "data/normalized_chatgpt_questions.json"
    clustered_train_path = "data/question_train_clustered_output.json"
    report_path = "samples/report/dataset_analysis_report.csv"

    # Analyze datasets
    train_report = analyze_dataset(train_path, "Training")
    test_report = analyze_dataset(test_path, "Test")
    clustered_report = analyze_dataset(clustered_train_path, "Clustered")
    augmented_train_path_report = analyze_dataset(augmented_train_path, "Augmented Training")

    # Combine results
    report = pd.DataFrame([train_report, test_report, augmented_train_path_report, clustered_report])

    # Save the report to a CSV file
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    report.to_csv(report_path, index=False)

    print(f"\nAnalysis report saved to {report_path}")