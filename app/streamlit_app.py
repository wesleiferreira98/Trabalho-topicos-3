import sys
from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / "src"))

from iot_fingerprint.data import load_processed_training
from iot_fingerprint.features import build_device_windows


st.set_page_config(page_title="IoT Fingerprint", layout="wide")

st.title("IoT Fingerprint")

df = load_processed_training()

st.metric("Frames tratados", f"{len(df):,}")
st.metric("Dispositivos de origem", df["wlan.sa"].nunique())

st.subheader("Top MACs de origem")
st.dataframe(df["wlan.sa"].value_counts().head(20).rename("frames"))

st.subheader("Distribuicao de Power Management")
st.bar_chart(df["wlan.fc.pwrmgt"].value_counts().sort_index())

st.subheader("Features por janela")
window_size = st.slider("Pacotes por janela", min_value=20, max_value=500, value=100, step=20)
features = build_device_windows(df, window_size=window_size)
st.dataframe(features.head(100))

