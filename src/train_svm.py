from iot_fingerprint.model_pipeline import run_training_pipeline
from model_specs import SVM_SPEC


def main() -> None:
    run_training_pipeline(SVM_SPEC)


if __name__ == "__main__":
    main()