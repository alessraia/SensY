from experiment_runner_ML import run_lazy

CONFIG = {
    "evaluation_mode": "lazy",
    "train_path": "data/dataset_SensY2.0.json",
    "random_state": 42,
    "lazy_test_size": 0.2,

    "report_dir": "samples/report",
    "model_dir": "samples/models",
    "errors_dir": "samples/errors",
    "results_dir": "samples/results",

    "experiment_tag": "sensy_lazy_seed42"
}

if __name__ == "__main__":
    run_lazy(CONFIG)