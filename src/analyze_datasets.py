from iot_fingerprint.data import load_processed_training


def main() -> None:
    df = load_processed_training()

    print("Dataset: data/02 - Tratados/processed_training2.csv")
    print(f"Linhas: {len(df):,}")
    print(f"Colunas: {list(df.columns)}")
    print()
    print("Top MACs de origem:")
    print(df["wlan.sa"].value_counts().head(15))
    print()
    print("Power management:")
    print(df["wlan.fc.pwrmgt"].value_counts())
    print()
    print("Resumo numerico:")
    print(df[["frame.time_delta_displayed", "frame.len"]].describe())


if __name__ == "__main__":
    main()

