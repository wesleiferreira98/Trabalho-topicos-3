from iot_fingerprint.model_pipeline import run_training_pipeline
from iot_fingerprint.config import REPORTS_DIR
from model_specs import ALL_SPECS, RF_SPEC


def train_all_models() -> None:
    for spec in ALL_SPECS:
        print(f"Training {spec.name}...")
        if spec.slug == RF_SPEC.slug:
            run_training_pipeline(spec, benchmark_specs=ALL_SPECS)
        else:
            run_training_pipeline(spec)

    print("\nTraining complete.")
    print("Main artifacts generated:")
    print(f"- Consolidated report: {REPORTS_DIR / 'model_report.md'}")
    print(f"- Academic report: {REPORTS_DIR / 'academic_report.md'}")
    print(f"- Model benchmark: {REPORTS_DIR / 'model_benchmark.csv'}")
    print(f"- Stratified CV results: {REPORTS_DIR / 'cv_results.csv'}")
    print(f"- Temporal CV results: {REPORTS_DIR / 'temporal_cv_results.csv'}")
    print(f"- Sliding temporal CV results: {REPORTS_DIR / 'sliding_temporal_cv_results.csv'}")


if __name__ == "__main__":
    train_all_models()