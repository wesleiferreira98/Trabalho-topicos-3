from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from joblib import dump
from sklearn.base import clone
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold

from iot_fingerprint.config import FIGURES_DIR, MODELS_DIR, REPORTS_DIR
from iot_fingerprint.data import load_processed_training
from iot_fingerprint.features import build_device_windows


MIN_WINDOWS_PER_DEVICE = 10
TEST_FRACTION = 0.2
CV_FOLDS = 5


@dataclass(frozen=True)
class ModelSpec:
    name: str
    slug: str
    estimator_factory: Callable[[], object]
    supports_feature_importance: bool = False
    param_distributions: dict[str, list | tuple] | None = None
    tuning_iterations: int = 10


def prepare_features() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = load_processed_training()
    features = build_device_windows(df)

    counts = features["wlan.sa"].value_counts()
    valid_devices = counts[counts >= MIN_WINDOWS_PER_DEVICE].index
    features = features[features["wlan.sa"].isin(valid_devices)].copy()

    train_df, test_df = temporal_device_split(features)
    return df, features, train_df, test_df, features.drop(columns=["wlan.sa", "_window"])


def temporal_device_split(
    features: pd.DataFrame,
    test_fraction: float = TEST_FRACTION,
) -> tuple[pd.DataFrame, pd.DataFrame]:
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


def stratified_cv(
    features: pd.DataFrame,
    estimator: object,
    n_splits: int = CV_FOLDS,
) -> pd.DataFrame:
    x = features.drop(columns=["wlan.sa", "_window"])
    y = features["wlan.sa"]
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_metrics: list[dict[str, float | int]] = []

    for fold, (train_idx, test_idx) in enumerate(splitter.split(x, y), start=1):
        fold_estimator = clone(estimator)
        x_train = x.iloc[train_idx]
        y_train = y.iloc[train_idx]
        x_test = x.iloc[test_idx]
        y_test = y.iloc[test_idx]

        fold_estimator.fit(x_train, y_train)
        predictions = fold_estimator.predict(x_test)

        fold_metrics.append(
            {
                "fold": fold,
                "train_windows": len(train_idx),
                "test_windows": len(test_idx),
                "accuracy": accuracy_score(y_test, predictions),
                "macro_f1": f1_score(y_test, predictions, average="macro"),
                "weighted_f1": f1_score(y_test, predictions, average="weighted"),
            }
        )

    return pd.DataFrame(fold_metrics)


def benchmark_models(
    model_specs: list[ModelSpec],
    features: pd.DataFrame,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_test: pd.DataFrame,
    y_test: pd.Series,
) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []

    for spec in model_specs:
        estimator, best_params, best_cv_score = tune_estimator(spec, x_train, y_train)
        holdout_metrics = evaluate_holdout(estimator, x_train, y_train, x_test, y_test)
        cv_results = stratified_cv(features, estimator)
        rows.append(
            {
                "model": spec.name,
                "holdout_accuracy": holdout_metrics["accuracy"],
                "holdout_macro_f1": holdout_metrics["macro_f1"],
                "holdout_weighted_f1": holdout_metrics["weighted_f1"],
                "cv_accuracy_mean": cv_results["accuracy"].mean(),
                "cv_macro_f1_mean": cv_results["macro_f1"].mean(),
                "cv_weighted_f1_mean": cv_results["weighted_f1"].mean(),
                "tuning_macro_f1": best_cv_score,
                "best_params": json.dumps(best_params, ensure_ascii=False, sort_keys=True) if best_params else "{}",
            }
        )

    return pd.DataFrame(rows).sort_values(
        by=["cv_macro_f1_mean", "cv_accuracy_mean"],
        ascending=False,
    )


def evaluate_holdout(
    estimator: object,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict[str, float]:
    estimator.fit(x_train, y_train)
    predictions = estimator.predict(x_test)
    return {
        "accuracy": accuracy_score(y_test, predictions),
        "macro_f1": f1_score(y_test, predictions, average="macro"),
        "weighted_f1": f1_score(y_test, predictions, average="weighted"),
    }


def tune_estimator(
    model_spec: ModelSpec,
    x_train: pd.DataFrame,
    y_train: pd.Series,
) -> tuple[object, dict[str, object], float] | tuple[object, None, None]:
    estimator = model_spec.estimator_factory()
    if not model_spec.param_distributions:
        estimator.fit(x_train, y_train)
        return estimator, None, None

    splitter = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=42)
    search = RandomizedSearchCV(
        estimator=estimator,
        param_distributions=model_spec.param_distributions,
        n_iter=model_spec.tuning_iterations,
        scoring="f1_macro",
        cv=splitter,
        random_state=42,
        n_jobs=-1,
        refit=True,
    )
    search.fit(x_train, y_train)
    return search.best_estimator_, search.best_params_, float(search.best_score_)


def run_training_pipeline(
    model_spec: ModelSpec,
    benchmark_specs: list[ModelSpec] | None = None,
) -> None:
    sns.set_theme(style="whitegrid")
    df, features, train_df, test_df, _ = prepare_features()
    x_train = train_df.drop(columns=["wlan.sa", "_window"])
    y_train = train_df["wlan.sa"]
    x_test = test_df.drop(columns=["wlan.sa", "_window"])
    y_test = test_df["wlan.sa"]

    estimator, best_params, best_cv_score = tune_estimator(model_spec, x_train, y_train)
    predictions = estimator.predict(x_test)

    print(
        f"Treino temporal por dispositivo: {len(x_train)} janelas de treino, "
        f"{len(x_test)} janelas de teste"
    )
    print(classification_report(y_test, predictions))
    print("Matriz de confusao:")
    print(confusion_matrix(y_test, predictions, labels=list(estimator.classes_)))
    if best_params is not None:
        print("Melhores hiperparametros:")
        print(best_params)
        print(f"Melhor macro F1 medio na busca: {best_cv_score:.4f}")

    importances = None
    if model_spec.supports_feature_importance:
        importances = pd.Series(estimator.feature_importances_, index=x_train.columns)
        print("Top 5 features mais importantes:")
        print(importances.sort_values(ascending=False).head(5))

    cv_results = stratified_cv(features, estimator)
    print("Validacao cruzada estratificada:")
    print(cv_results.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print("Media CV estratificada:")
    print(
        cv_results[["accuracy", "macro_f1", "weighted_f1"]]
        .mean()
        .to_string(float_format=lambda value: f"{value:.4f}")
    )

    benchmark_df = None
    if benchmark_specs:
        benchmark_df = benchmark_models(benchmark_specs, features, x_train, y_train, x_test, y_test)
        print("Comparacao entre modelos:")
        print(benchmark_df.to_string(index=False, float_format=lambda value: f"{value:.4f}"))

    save_confusion_matrix_figure(model_spec, y_test, pd.Series(predictions), list(estimator.classes_))
    if importances is not None:
        save_feature_importance_figure(model_spec, importances)
    save_cv_metrics_figure(model_spec, cv_results)
    save_model_outputs(
        model_spec,
        estimator,
        cv_results,
        benchmark_df,
        df,
        features,
        importances,
        y_test,
        pd.Series(predictions),
        best_params,
        best_cv_score,
    )


def save_model_outputs(
    model_spec: ModelSpec,
    estimator: object,
    cv_results: pd.DataFrame,
    benchmark_df: pd.DataFrame | None,
    df: pd.DataFrame,
    features: pd.DataFrame,
    importances: pd.Series | None,
    y_test: pd.Series,
    predictions: pd.Series,
    best_params: dict[str, object] | None,
    best_cv_score: float | None,
) -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    dump(estimator, MODELS_DIR / f"{model_spec.slug}_device_fingerprint.joblib")
    cv_results.to_csv(REPORTS_DIR / f"{model_spec.slug}_cv_results.csv", index=False)
    if best_params is not None:
        tuning_payload = {
            "model": model_spec.name,
            "best_params": best_params,
            "best_macro_f1_cv": best_cv_score,
        }
        (REPORTS_DIR / f"{model_spec.slug}_best_params.json").write_text(
            json.dumps(tuning_payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    if model_spec.slug == "rf":
        cv_results.to_csv(REPORTS_DIR / "cv_results.csv", index=False)
    if benchmark_df is not None:
        benchmark_df.to_csv(REPORTS_DIR / "model_benchmark.csv", index=False)
        save_model_comparison_figure(benchmark_df)
    write_consolidated_report(
        model_spec,
        df,
        features,
        importances,
        cv_results,
        benchmark_df,
        y_test,
        predictions,
        best_params,
        best_cv_score,
    )


def save_confusion_matrix_figure(
    model_spec: ModelSpec,
    y_true: pd.Series,
    y_pred: pd.Series,
    labels: list[str],
) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(11, 9))
    sns.heatmap(cm, cmap="Blues", annot=True, fmt="d", xticklabels=labels, yticklabels=labels)
    plt.title(f"Matriz de confusao do {model_spec.name}")
    plt.xlabel("Predito")
    plt.ylabel("Real")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f"{model_spec.slug}_confusion_matrix.png", dpi=200)
    plt.close()


def save_feature_importance_figure(model_spec: ModelSpec, importances: pd.Series) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    ordered = importances.sort_values(ascending=True)
    plt.figure(figsize=(10, 6))
    ordered.plot(kind="barh", color="#2a9d8f")
    plt.title(f"Importancia das features no {model_spec.name}")
    plt.xlabel("Importancia")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f"{model_spec.slug}_feature_importance.png", dpi=200)
    plt.close()


def save_cv_metrics_figure(model_spec: ModelSpec, cv_results: pd.DataFrame) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    melted = cv_results.melt(
        id_vars=["fold"],
        value_vars=["accuracy", "macro_f1", "weighted_f1"],
        var_name="metric",
        value_name="value",
    )
    plt.figure(figsize=(9, 5))
    sns.lineplot(data=melted, x="fold", y="value", hue="metric", marker="o")
    plt.title(f"Evolucao das metricas por fold do {model_spec.name}")
    plt.xlabel("Fold")
    plt.ylabel("Valor da metrica")
    plt.ylim(0.85, 1.01)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f"{model_spec.slug}_cv_metrics_by_fold.png", dpi=200)
    plt.close()


def save_model_comparison_figure(benchmark_df: pd.DataFrame) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    melted = benchmark_df.melt(
        id_vars=["model"],
        value_vars=["cv_accuracy_mean", "cv_macro_f1_mean", "cv_weighted_f1_mean"],
        var_name="metric",
        value_name="value",
    )
    plt.figure(figsize=(10, 6))
    sns.barplot(data=melted, x="model", y="value", hue="metric")
    plt.title("Comparacao entre modelos na CV estratificada")
    plt.xlabel("Modelo")
    plt.ylabel("Valor medio")
    plt.ylim(0.85, 1.01)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "model_comparison_cv.png", dpi=200)
    plt.close()


def write_consolidated_report(
    model_spec: ModelSpec,
    df: pd.DataFrame,
    features: pd.DataFrame,
    importances: pd.Series | None,
    cv_results: pd.DataFrame,
    benchmark_df: pd.DataFrame | None,
    y_test: pd.Series,
    predictions: pd.Series,
    best_params: dict[str, object] | None,
    best_cv_score: float | None,
) -> None:
    classification_text = classification_report(y_test, predictions)
    eda_summary_path = FIGURES_DIR / "eda_summary.txt"
    eda_summary = ""
    if eda_summary_path.exists():
        eda_summary = eda_summary_path.read_text(encoding="utf-8").strip()

    report_lines = [
        f"# Relatorio Consolidado - {model_spec.name}",
        "",
        "## Resumo Executivo",
        f"- Linhas tratadas: {len(df)}",
        f"- Dispositivos de origem: {df['wlan.sa'].nunique()}",
        f"- Janelas usadas no modelo: {len(features)}",
        f"- Acuracia media da CV estratificada: {cv_results['accuracy'].mean():.4f}",
        f"- Macro F1 medio da CV estratificada: {cv_results['macro_f1'].mean():.4f}",
        "",
        "## EDA",
    ]
    if eda_summary:
        report_lines.extend(["```text", eda_summary, "```", ""])
    report_lines.extend(
        [
            "## Treino e Validacao",
            "### Holdout temporal por dispositivo",
            "```text",
            classification_text.strip(),
            "```",
            "",
            "### Validacao cruzada estratificada",
            cv_results_to_markdown(cv_results),
            "",
        ]
    )
    if best_params is not None:
        report_lines.extend(
            [
                "### Hiperparametros selecionados automaticamente",
                f"- Melhor macro F1 medio na busca: {best_cv_score:.4f}",
            ]
        )
        report_lines.extend(f"- {name}: {value}" for name, value in best_params.items())
        report_lines.append("")
    if benchmark_df is not None:
        report_lines.extend(["### Comparacao entre algoritmos", benchmark_to_markdown(benchmark_df), ""])
    if importances is not None:
        report_lines.append("### Features mais importantes")
        report_lines.extend(
            f"- {name}: {value:.4f}"
            for name, value in importances.sort_values(ascending=False).items()
        )
        report_lines.append("")
    report_lines.extend(
        [
            "## Artefatos do Modelo",
            f"- reports/figures/{model_spec.slug}_confusion_matrix.png",
            f"- reports/figures/{model_spec.slug}_cv_metrics_by_fold.png",
            f"- reports/{model_spec.slug}_cv_results.csv",
            f"- models/{model_spec.slug}_device_fingerprint.joblib",
            "",
        ]
    )
    if importances is not None:
        insert_at = len(report_lines) - 1
        report_lines.insert(insert_at, f"- reports/figures/{model_spec.slug}_feature_importance.png")
    if benchmark_df is not None:
        report_lines.insert(len(report_lines) - 1, "- reports/model_benchmark.csv")
        report_lines.insert(len(report_lines) - 1, "- reports/figures/model_comparison_cv.png")
    if best_params is not None:
        report_lines.insert(len(report_lines) - 1, f"- reports/{model_spec.slug}_best_params.json")

    (REPORTS_DIR / f"{model_spec.slug}_model_report.md").write_text("\n".join(report_lines), encoding="utf-8")
    if model_spec.slug == "rf":
        (REPORTS_DIR / "model_report.md").write_text("\n".join(report_lines), encoding="utf-8")


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


def benchmark_to_markdown(benchmark_df: pd.DataFrame) -> str:
    header = (
        "| model | holdout_accuracy | holdout_macro_f1 | holdout_weighted_f1 | "
        "cv_accuracy_mean | cv_macro_f1_mean | cv_weighted_f1_mean | tuning_macro_f1 | best_params |"
    )
    separator = "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"
    rows = [header, separator]
    for row in benchmark_df.itertuples(index=False):
        rows.append(
            "| "
            f"{row.model} | {row.holdout_accuracy:.4f} | {row.holdout_macro_f1:.4f} | "
            f"{row.holdout_weighted_f1:.4f} | {row.cv_accuracy_mean:.4f} | "
            f"{row.cv_macro_f1_mean:.4f} | {row.cv_weighted_f1_mean:.4f} | {row.tuning_macro_f1:.4f} | {row.best_params} |"
        )
    return "\n".join(rows)