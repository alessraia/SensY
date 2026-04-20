from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

def train_model(features, labels, split=True):
    """
    Trains a Random Forest model with parallelization.
    If split=True, performs an internal split on features and labels.
    If split=False, trains directly on features and labels and returns None for X_test, y_test.
    """
    if split:
        X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
    else:
        X_train, y_train = features, labels
        X_test, y_test = None, None

    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    return model, X_test, y_test