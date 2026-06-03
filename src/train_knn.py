from iot_fingerprint.model_pipeline import run_training_pipeline
from model_specs import KNN_SPEC


def main() -> None:
    run_training_pipeline(KNN_SPEC)


if __name__ == "__main__":
    main()