from iot_fingerprint.model_pipeline import run_training_pipeline
from model_specs import ALL_SPECS, RF_SPEC


def main() -> None:
    run_training_pipeline(RF_SPEC, benchmark_specs=ALL_SPECS)


if __name__ == "__main__":
    main()

