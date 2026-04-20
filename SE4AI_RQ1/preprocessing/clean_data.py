import pandas as pd
import nltk
import json

def clean_dataset(file_path):
    """
    Load and clean the dataset from a JSON file.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    df = pd.DataFrame(data)
    print(f"Before cleaning {file_path}: {df.count()}")

    # Remove duplicates
    #df = df.drop_duplicates()
    #print(f"After cleaning {file_path}: {df.count()}")

    # If the 'question' column exists, remove it
    if 'question' in df.columns:
        df = df.drop(columns=['question'])

    # Tokenization of English text
    df["question_tokenized"] = df["question_en"].apply(nltk.word_tokenize)

    return df