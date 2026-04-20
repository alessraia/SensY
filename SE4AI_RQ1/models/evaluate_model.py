from sklearn.metrics import classification_report

def evaluate_model(model, X_test, y_test, print_report=True):
    predictions = model.predict(X_test)
    report = classification_report(y_test, predictions, output_dict=True)
    if print_report:
        print("Report:")
        print(report)
    return report