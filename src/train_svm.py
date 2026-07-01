from iot_fingerprint.model_pipeline import run_training_pipeline
from model_specs import SVM_SPEC


def main() -> None:
    """Executa o pipeline completo do SVM com kernel RBF e normalizacao via StandardScaler integrado ao Pipeline do scikit-learn."""
    run_training_pipeline(SVM_SPEC)


if __name__ == "__main__":
    main()