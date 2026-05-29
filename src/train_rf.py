import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from joblib import dump
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from iot_fingerprint.config import FIGURES_DIR, MODELS_DIR, REPORTS_DIR
from iot_fingerprint.data import load_processed_training
from iot_fingerprint.features import build_device_windows


MIN_WINDOWS_PER_DEVICE = 10
TEST_FRACTION = 0.2
CV_BLOCKS = 5


def temporal_device_split(
    features: pd.DataFrame,
    test_fraction: float = TEST_FRACTION,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split each device chronologically to avoid leakage across adjacent windows."""
    train_parts: list[pd.DataFrame] = []
    test_parts: list[pd.DataFrame] = []

    for _, device_rows in features.groupby("wlan.sa", sort=False):
        ordered_rows = device_rows.sort_values("_window").reset_index(drop=True)
        split_idx = max(1, int(len(ordered_rows) * (1 - test_fraction)))
        split_idx = min(split_idx, len(ordered_rows) - 1)
        train_parts.append(ordered_rows.iloc[:split_idx])
        test_parts.append(ordered_rows.iloc[split_idx:])

    train_df = pd.concat(train_parts, ignore_index=True)
    test_df = pd.concat(test_parts, ignore_index=True)
    return train_df, test_df


def add_temporal_blocks(features: pd.DataFrame, n_blocks: int = CV_BLOCKS) -> pd.DataFrame:
    """Assign contiguous temporal blocks per device for walk-forward validation."""
    blocked_parts: list[pd.DataFrame] = []

    for _, device_rows in features.groupby("wlan.sa", sort=False):
        ordered_rows = device_rows.sort_values("_window").reset_index(drop=True).copy()
        ordered_rows["_cv_block"] = (
            pd.Series(range(len(ordered_rows))) * n_blocks // len(ordered_rows)
        ).clip(upper=n_blocks - 1)
        blocked_parts.append(ordered_rows)

    return pd.concat(blocked_parts, ignore_index=True)


def temporal_block_cv(
    features: pd.DataFrame,
    n_blocks: int = CV_BLOCKS,
) -> pd.DataFrame:
    """Evaluate the model with walk-forward temporal blocks per device."""
    blocked = add_temporal_blocks(features, n_blocks=n_blocks)
    fold_metrics: list[dict[str, float | int]] = []

    for fold in range(1, n_blocks):
        train_df = blocked[blocked["_cv_block"] < fold]
        test_df = blocked[blocked["_cv_block"] == fold]

        x_train = train_df.drop(columns=["wlan.sa", "_window", "_cv_block"])
        y_train = train_df["wlan.sa"]
        x_test = test_df.drop(columns=["wlan.sa", "_window", "_cv_block"])
        y_test = test_df["wlan.sa"]

        model = RandomForestClassifier(
            n_estimators=300,
            random_state=42,
            class_weight="balanced",
            n_jobs=-1,
        )
        model.fit(x_train, y_train)
        predictions = model.predict(x_test)

        fold_metrics.append(
            {
                "fold": fold,
                "train_windows": len(train_df),
                "test_windows": len(test_df),
                "accuracy": accuracy_score(y_test, predictions),
                "macro_f1": f1_score(y_test, predictions, average="macro"),
                "weighted_f1": f1_score(y_test, predictions, average="weighted"),
            }
        )

    return pd.DataFrame(fold_metrics)


def save_confusion_matrix_figure(
    y_true: pd.Series,
    y_pred: pd.Series,
    labels: list[str],
) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(11, 9))
    sns.heatmap(cm, cmap="Blues", annot=True, fmt="d", xticklabels=labels, yticklabels=labels)
    plt.title("Matriz de confusao do Random Forest")
    plt.xlabel("Predito")
    plt.ylabel("Real")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "rf_confusion_matrix.png", dpi=200)
    plt.close()


def save_feature_importance_figure(importances: pd.Series) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    ordered = importances.sort_values(ascending=True)
    plt.figure(figsize=(10, 6))
    ordered.plot(kind="barh", color="#2a9d8f")
    plt.title("Importancia das features no Random Forest")
    plt.xlabel("Importancia")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "rf_feature_importance.png", dpi=200)
    plt.close()


def save_cv_metrics_figure(cv_results: pd.DataFrame) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    melted = cv_results.melt(
        id_vars=["fold"],
        value_vars=["accuracy", "macro_f1", "weighted_f1"],
        var_name="metric",
        value_name="value",
    )
    plt.figure(figsize=(9, 5))
    sns.lineplot(data=melted, x="fold", y="value", hue="metric", marker="o")
    plt.title("Evolucao das metricas por fold da CV temporal")
    plt.xlabel("Fold")
    plt.ylabel("Valor da metrica")
    plt.ylim(0.85, 1.01)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "rf_cv_metrics_by_fold.png", dpi=200)
    plt.close()


def cv_results_to_markdown(cv_results: pd.DataFrame) -> str:
    header = "| fold | train_windows | test_windows | accuracy | macro_f1 | weighted_f1 |"
    separator = "| --- | --- | --- | --- | --- | --- |"
    rows = [header, separator]
    for row in cv_results.itertuples(index=False):
        rows.append(
            "| "
            f"{row.fold} | {row.train_windows} | {row.test_windows} | "
            f"{row.accuracy:.4f} | {row.macro_f1:.4f} | {row.weighted_f1:.4f} |"
        )
    return "\n".join(rows)


def write_consolidated_report(
    df: pd.DataFrame,
    features: pd.DataFrame,
    importances: pd.Series,
    cv_results: pd.DataFrame,
    y_test: pd.Series,
    predictions: pd.Series,
) -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    classification_text = classification_report(y_test, predictions)
    eda_summary_path = FIGURES_DIR / "eda_summary.txt"
    eda_summary = ""
    if eda_summary_path.exists():
        eda_summary = eda_summary_path.read_text(encoding="utf-8").strip()

    report_lines = [
        "# Relatorio Consolidado - IoT Fingerprint",
        "",
        "## Resumo Executivo",
        f"- Linhas tratadas: {len(df)}",
        f"- Dispositivos de origem: {df['wlan.sa'].nunique()}",
        f"- Janelas usadas no modelo: {len(features)}",
        f"- Acuracia media da CV temporal: {cv_results['accuracy'].mean():.4f}",
        f"- Macro F1 medio da CV temporal: {cv_results['macro_f1'].mean():.4f}",
        "",
        "## EDA",
    ]
    if eda_summary:
        report_lines.extend(["```text", eda_summary, "```", ""])
    report_lines.extend(
        [
            "Figuras geradas:",
            "- reports/figures/top_sources.png",
            "- reports/figures/power_management_distribution.png",
            "- reports/figures/frame_length_by_device.png",
            "- reports/figures/window_feature_scatter.png",
            "- reports/figures/feature_correlation_heatmap.png",
            "",
            "## Treino e Validacao",
            "### Holdout temporal por dispositivo",
            "```text",
            classification_text.strip(),
            "```",
            "",
            "### Validacao cruzada temporal por blocos",
            cv_results_to_markdown(cv_results),
            "",
            "### Features mais importantes",
        ]
    )
    report_lines.extend(
        f"- {name}: {value:.4f}"
        for name, value in importances.sort_values(ascending=False).items()
    )
    report_lines.extend(
        [
            "",
            "## Artefatos do Modelo",
            "- reports/figures/rf_confusion_matrix.png",
            "- reports/figures/rf_feature_importance.png",
            "- reports/figures/rf_cv_metrics_by_fold.png",
            "- reports/cv_results.csv",
            "- models/random_forest_device_fingerprint.joblib",
            "",
        ]
    )
    (REPORTS_DIR / "model_report.md").write_text("\n".join(report_lines), encoding="utf-8")


def main() -> None:
    sns.set_theme(style="whitegrid")
    df = load_processed_training()
    features = build_device_windows(df)

    counts = features["wlan.sa"].value_counts()
    valid_devices = counts[counts >= MIN_WINDOWS_PER_DEVICE].index
    features = features[features["wlan.sa"].isin(valid_devices)].copy()

    train_df, test_df = temporal_device_split(features)
    x_train = train_df.drop(columns=["wlan.sa", "_window"])
    y_train = train_df["wlan.sa"]
    x_test = test_df.drop(columns=["wlan.sa", "_window"])
    y_test = test_df["wlan.sa"]

    model = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1,
    )
    model.fit(x_train, y_train)

    predictions = model.predict(x_test)
    print(
        f"Treino temporal por dispositivo: {len(x_train)} janelas de treino, "
        f"{len(x_test)} janelas de teste"
    )
    print(classification_report(y_test, predictions))
    print("Matriz de confusao:")
    print(confusion_matrix(y_test, predictions, labels=model.classes_))
    importances = pd.Series(model.feature_importances_, index=x_train.columns)
    print("Top 5 features mais importantes:")
    print(importances.sort_values(ascending=False).head(5))

    cv_results = temporal_block_cv(features)
    print("Validacao cruzada temporal por blocos:")
    print(cv_results.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print("Media CV temporal:")
    print(
        cv_results[["accuracy", "macro_f1", "weighted_f1"]]
        .mean()
        .to_string(float_format=lambda value: f"{value:.4f}")
    )

    save_confusion_matrix_figure(y_test, predictions, list(model.classes_))
    save_feature_importance_figure(importances)
    save_cv_metrics_figure(cv_results)
    cv_results.to_csv(REPORTS_DIR / "cv_results.csv", index=False)
    write_consolidated_report(df, features, importances, cv_results, y_test, predictions)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    dump(model, MODELS_DIR / "random_forest_device_fingerprint.joblib")


if __name__ == "__main__":
    main()

