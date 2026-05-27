from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "01 - Não-Tratados"
PROCESSED_DATA_DIR = DATA_DIR / "02 - Tratados"
FEATURES_DATA_DIR = DATA_DIR / "03 - Features"

PROCESSED_TRAINING_CSV = PROCESSED_DATA_DIR / "processed_training2.csv"

MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

