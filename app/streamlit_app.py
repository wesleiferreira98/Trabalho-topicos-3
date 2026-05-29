import sys
from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / "src"))

REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

from iot_fingerprint.data import load_processed_training
from iot_fingerprint.features import build_device_windows


st.set_page_config(page_title="IoT Fingerprint", layout="wide")

st.title("IoT Fingerprint")


@st.cache_data
def load_cv_results() -> pd.DataFrame | None:
	path = REPORTS_DIR / "cv_results.csv"
	if not path.exists():
		return None
	return pd.read_csv(path)


def read_report_markdown() -> str | None:
	path = REPORTS_DIR / "model_report.md"
	if not path.exists():
		return None
	return path.read_text(encoding="utf-8")


def show_figure(path: Path, caption: str) -> None:
	if path.exists():
		st.image(str(path), caption=caption, width="stretch")
	else:
		st.info(f"Figura ainda nao gerada: {path.name}")

df = load_processed_training()
cv_results = load_cv_results()
report_markdown = read_report_markdown()

if cv_results is not None:
	metric_cols = st.columns(4)
	metric_cols[0].metric("Frames tratados", f"{len(df):,}")
	metric_cols[1].metric("Dispositivos de origem", df["wlan.sa"].nunique())
	metric_cols[2].metric("Accuracy media CV", f"{cv_results['accuracy'].mean():.4f}")
	metric_cols[3].metric("Macro F1 medio CV", f"{cv_results['macro_f1'].mean():.4f}")
else:
	metric_cols = st.columns(2)
	metric_cols[0].metric("Frames tratados", f"{len(df):,}")
	metric_cols[1].metric("Dispositivos de origem", df["wlan.sa"].nunique())

overview_tab, eda_tab, model_tab, report_tab = st.tabs(
	["Visao Geral", "EDA", "Modelo", "Relatorio"]
)

with overview_tab:
	left_col, right_col = st.columns([1, 1])
	with left_col:
		st.subheader("Top MACs de origem")
		st.dataframe(df["wlan.sa"].value_counts().head(20).rename("frames"))

		st.subheader("Distribuicao de Power Management")
		st.bar_chart(df["wlan.fc.pwrmgt"].value_counts().sort_index())

	with right_col:
		st.subheader("Features por janela")
		window_size = st.slider("Pacotes por janela", min_value=20, max_value=500, value=100, step=20)
		features = build_device_windows(df, window_size=window_size)
		st.dataframe(features.head(100), width="stretch")

with eda_tab:
	st.subheader("Figuras exploratorias")
	show_figure(FIGURES_DIR / "top_sources.png", "Top 15 dispositivos por frames")
	show_figure(FIGURES_DIR / "power_management_distribution.png", "Distribuicao de power management")
	show_figure(FIGURES_DIR / "frame_length_by_device.png", "Distribuicao do tamanho dos frames")
	show_figure(FIGURES_DIR / "window_feature_scatter.png", "Separacao de dispositivos no espaco de features")
	show_figure(FIGURES_DIR / "feature_correlation_heatmap.png", "Correlacao entre features agregadas")

with model_tab:
	st.subheader("Validacao e interpretabilidade")
	if cv_results is not None:
		st.dataframe(cv_results, width="stretch")
		st.line_chart(cv_results.set_index("fold")[["accuracy", "macro_f1", "weighted_f1"]])
	else:
		st.info("Rode src/train_rf.py para gerar as metricas de validacao.")

	show_figure(FIGURES_DIR / "rf_confusion_matrix.png", "Matriz de confusao do modelo")
	show_figure(FIGURES_DIR / "rf_feature_importance.png", "Importancia das features")
	show_figure(FIGURES_DIR / "rf_cv_metrics_by_fold.png", "Evolucao das metricas por fold")

with report_tab:
	st.subheader("Relatorio consolidado")
	if report_markdown:
		st.markdown(report_markdown)
	else:
		st.info("Rode src/train_rf.py para gerar reports/model_report.md.")

