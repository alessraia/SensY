from experiment_runner_ML import run_selected_models

CONFIG = {
    "evaluation_mode": "cv10",

    "selected_models": [
        "dummy_most_frequent",
        "logistic_regression",
        "linear_svc",
        "svc",
        "ridge_classifier",
        "sgd_classifier",
        "random_forest",
        "extra_trees"
    ],

    "train_path": "data/dataset_SensY2.0.json",
    "test_path": "data/dataset_SensY2.0.json",

    "random_state": 42,
    "lazy_test_size": 0.2,

    "report_dir": "samples/report",
    "model_dir": "samples/models",
    "errors_dir": "samples/errors",
    "results_dir": "samples/results",

    "experiment_tag": "ml_shortlist_cv10_sensy2"
}

if __name__ == "__main__":
    run_selected_models(CONFIG)