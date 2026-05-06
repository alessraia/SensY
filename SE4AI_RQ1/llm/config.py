from pathlib import Path

BASE_DIR = Path(r"D:\SE4AI_RQ1_LLM")
CHECKPOINTS_DIR = BASE_DIR / "checkpoints"
OUTPUTS_DIR = BASE_DIR / "outputs"
LOGS_DIR = BASE_DIR / "logs"
TEMP_DIR = BASE_DIR / "temp"

PROJECT_ROOT = Path(r"C:\Users\marty\PycharmProjects\SensY\SE4AI_RQ1")
DATA_DIR = PROJECT_ROOT / "data"

MODEL_CONFIGS = {
    "gemma": {
        "model_id": "google/gemma-3-4b-it",
        "output_dir": CHECKPOINTS_DIR / "gemma_4b",
        "logs_dir": LOGS_DIR / "gemma_4b",
    },
    "llama": {
        "model_id": "meta-llama/Llama-3.2-3B-Instruct",
        "output_dir": CHECKPOINTS_DIR / "llama_3b",
        "logs_dir": LOGS_DIR / "llama_3b",
    },
}

TRAIN_FILE = DATA_DIR / "dataset_SENSY2.0.json"
SQUARE_FILE = DATA_DIR / "dataset_SQUARE.json"

RANDOM_SEED = 42
TEST_SIZE = 0.2
MAX_SEQ_LENGTH = 256

NUM_TRAIN_EPOCHS = 2
LEARNING_RATE = 2e-4
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.03

PER_DEVICE_TRAIN_BATCH_SIZE = 2
PER_DEVICE_EVAL_BATCH_SIZE = 2
GRADIENT_ACCUMULATION_STEPS = 8

LOGGING_STEPS = 10
SAVE_TOTAL_LIMIT = 2

LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05