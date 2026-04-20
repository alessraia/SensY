import pickle
import os
from extract_single import extract_features_single

if __name__ == "__main__":
    samples_dir = "samples/models"
    best_model_path = os.path.join(samples_dir, "model_square_testset_ratio_1.0.pkl")

    with open(best_model_path, "rb") as f:
        model = pickle.load(f)

    # ask a question to the user
    question = input("Enter your question (in English): ")

    # Extract features from the single question (only syntactic + BERT)
    features = extract_features_single(question)  # No need to pass vectorizers anymore

    # Prediction
    prediction = model.predict(features)

    # Print the result
    if prediction[0] == 1:
        print("The question is considered SENSITIVE.")
    else:
        print("The question is NOT considered SENSITIVE.")