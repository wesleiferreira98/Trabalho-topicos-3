import pandas as pd

from iot_fingerprint.config import PROCESSED_TRAINING_CSV


def load_processed_training(path=PROCESSED_TRAINING_CSV) -> pd.DataFrame:
    """Load the treated Wi-Fi frame dataset with normalized column types."""
    df = pd.read_csv(path)

    df["frame.time_delta_displayed"] = pd.to_numeric(
        df["frame.time_delta_displayed"],
        errors="coerce",
    )
    df["frame.len"] = pd.to_numeric(df["frame.len"], errors="coerce")
    df["wlan.fc.pwrmgt"] = df["wlan.fc.pwrmgt"].astype(str).str.lower().map(
        {"true": 1, "false": 0}
    )

    return df.dropna()

