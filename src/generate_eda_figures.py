from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from iot_fingerprint.config import FIGURES_DIR
from iot_fingerprint.data import load_processed_training
from iot_fingerprint.features import build_device_windows


MAX_POINTS_PER_DEVICE = 2000


def save_top_sources(df: pd.DataFrame, output_dir: Path) -> None:
    top_sources = df["wlan.sa"].value_counts().head(15).sort_values()
    plt.figure(figsize=(10, 7))
    top_sources.plot(kind="barh", color="#2a9d8f")
    plt.title("Top 15 dispositivos por quantidade de frames")
    plt.xlabel("Frames")
    plt.ylabel("MAC de origem")
    plt.tight_layout()
    plt.savefig(output_dir / "top_sources.png", dpi=200)
    plt.close()


def save_power_management(df: pd.DataFrame, output_dir: Path) -> None:
    power_counts = df["wlan.fc.pwrmgt"].value_counts().sort_index()
    labels = ["Desativado", "Ativado"]
    plt.figure(figsize=(7, 5))
    plt.bar(labels, power_counts.values, color=["#457b9d", "#e76f51"])
    plt.title("Distribuicao de power management")
    plt.ylabel("Frames")
    plt.tight_layout()
    plt.savefig(output_dir / "power_management_distribution.png", dpi=200)
    plt.close()


def save_frame_length_boxplot(df: pd.DataFrame, output_dir: Path) -> None:
    top_devices = df["wlan.sa"].value_counts().head(8).index
    sample = (
        df[df["wlan.sa"].isin(top_devices)]
        .groupby("wlan.sa", group_keys=False)
        .apply(
            lambda rows: rows.sample(min(len(rows), MAX_POINTS_PER_DEVICE), random_state=42).assign(
                device=rows.name
            )
        )
        .reset_index(drop=True)
        .rename(columns={"frame.len": "frame_len"})
    )

    plt.figure(figsize=(12, 6))
    sample.boxplot(column="frame_len", by="device", rot=35, grid=False)
    plt.title("Distribuicao do tamanho dos frames nos principais dispositivos")
    plt.xlabel("MAC de origem")
    plt.ylabel("Tamanho do frame")
    plt.suptitle("")
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    plt.savefig(output_dir / "frame_length_by_device.png", dpi=200)
    plt.close()


def save_window_scatter(features: pd.DataFrame, output_dir: Path) -> None:
    top_devices = features["wlan.sa"].value_counts().head(6).index
    sample = features[features["wlan.sa"].isin(top_devices)].copy()
    plt.figure(figsize=(10, 7))
    sns.scatterplot(
        data=sample,
        x="frame_len_mean",
        y="iat_mean",
        hue="wlan.sa",
        size="unique_destinations",
        alpha=0.7,
    )
    plt.title("Separacao de dispositivos no espaco de features")
    plt.xlabel("Media do tamanho do frame por janela")
    plt.ylabel("Media do IAT por janela")
    plt.tight_layout()
    plt.savefig(output_dir / "window_feature_scatter.png", dpi=200)
    plt.close()


def save_feature_correlation(features: pd.DataFrame, output_dir: Path) -> None:
    numeric_cols = [
        "packet_count",
        "frame_len_mean",
        "frame_len_std",
        "frame_len_min",
        "frame_len_max",
        "iat_mean",
        "iat_std",
        "iat_min",
        "iat_max",
        "pwrmgt_ratio",
        "unique_destinations",
    ]
    corr = features[numeric_cols].corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, cmap="crest", annot=True, fmt=".2f", square=True)
    plt.title("Correlacao entre features agregadas")
    plt.tight_layout()
    plt.savefig(output_dir / "feature_correlation_heatmap.png", dpi=200)
    plt.close()


def save_summary(df: pd.DataFrame, features: pd.DataFrame, output_dir: Path) -> None:
    summary_path = output_dir / "eda_summary.txt"
    top_sources = df["wlan.sa"].value_counts().head(10)
    summary_lines = [
        f"Linhas tratadas: {len(df)}",
        f"Dispositivos de origem: {df['wlan.sa'].nunique()}",
        f"Dispositivos com janelas suficientes: {features['wlan.sa'].nunique()}",
        f"Janelas agregadas: {len(features)}",
        f"Frame len medio: {df['frame.len'].mean():.3f}",
        f"IAT medio: {df['frame.time_delta_displayed'].mean():.6f}",
        "Top 10 MACs de origem:",
    ]
    summary_lines.extend(f"- {mac}: {count}" for mac, count in top_sources.items())
    summary_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")


def main() -> None:
    sns.set_theme(style="whitegrid")
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    df = load_processed_training()
    features = build_device_windows(df)
    counts = features["wlan.sa"].value_counts()
    features = features[features["wlan.sa"].isin(counts[counts >= 10].index)].copy()

    save_top_sources(df, FIGURES_DIR)
    save_power_management(df, FIGURES_DIR)
    save_frame_length_boxplot(df, FIGURES_DIR)
    save_window_scatter(features, FIGURES_DIR)
    save_feature_correlation(features, FIGURES_DIR)
    save_summary(df, features, FIGURES_DIR)

    print(f"Figuras salvas em: {FIGURES_DIR}")


if __name__ == "__main__":
    main()