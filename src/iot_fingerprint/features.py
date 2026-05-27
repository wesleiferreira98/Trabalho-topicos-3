import pandas as pd


def build_device_windows(
    df: pd.DataFrame,
    device_col: str = "wlan.sa",
    window_size: int = 100,
) -> pd.DataFrame:
    """Aggregate frame-level traffic into fixed-size packet windows per device."""
    work = df.copy()
    work["_window"] = work.groupby(device_col).cumcount() // window_size

    features = (
        work.groupby([device_col, "_window"])
        .agg(
            packet_count=("frame.len", "size"),
            frame_len_mean=("frame.len", "mean"),
            frame_len_std=("frame.len", "std"),
            frame_len_min=("frame.len", "min"),
            frame_len_max=("frame.len", "max"),
            iat_mean=("frame.time_delta_displayed", "mean"),
            iat_std=("frame.time_delta_displayed", "std"),
            iat_min=("frame.time_delta_displayed", "min"),
            iat_max=("frame.time_delta_displayed", "max"),
            pwrmgt_ratio=("wlan.fc.pwrmgt", "mean"),
            unique_destinations=("wlan.da", "nunique"),
        )
        .reset_index()
    )

    return features.fillna(0)

