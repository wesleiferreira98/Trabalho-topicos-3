import json
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
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


@st.cache_data
def load_temporal_cv_results() -> pd.DataFrame | None:
	path = REPORTS_DIR / "temporal_cv_results.csv"
	if not path.exists():
		return None
	return pd.read_csv(path)


@st.cache_data
def load_sliding_temporal_cv_results() -> pd.DataFrame | None:
	path = REPORTS_DIR / "sliding_temporal_cv_results.csv"
	if not path.exists():
		return None
	return pd.read_csv(path)


@st.cache_data
def load_model_benchmark() -> pd.DataFrame | None:
	path = REPORTS_DIR / "model_benchmark.csv"
	if not path.exists():
		return None
	benchmark_df = pd.read_csv(path)
	if "best_params" in benchmark_df.columns:
		benchmark_df["best_params"] = benchmark_df["best_params"].apply(
			lambda value: json.loads(value) if isinstance(value, str) and value else {}
		)
	return benchmark_df


def load_report_markdowns() -> dict[str, str]:
	reports = {
		"Consolidado": REPORTS_DIR / "model_report.md",
		"Academico": REPORTS_DIR / "academic_report.md",
	}
	loaded_reports: dict[str, str] = {}
	for label, path in reports.items():
		if path.exists():
			loaded_reports[label] = path.read_text(encoding="utf-8")
	return loaded_reports


def show_figure(path: Path, caption: str) -> None:
	if path.exists():
		st.image(str(path), caption=caption, width="stretch")
	else:
		st.info(f"Figura ainda nao gerada: {path.name}")

df = load_processed_training()
cv_results = load_cv_results()
temporal_cv_results = load_temporal_cv_results()
sliding_temporal_cv_results = load_sliding_temporal_cv_results()
benchmark_df = load_model_benchmark()
report_markdowns = load_report_markdowns()

if cv_results is not None:
	metric_cols = st.columns(5)
	metric_cols[0].metric("Frames tratados", f"{len(df):,}")
	metric_cols[1].metric("Dispositivos de origem", df["wlan.sa"].nunique())
	metric_cols[2].metric("Accuracy media CV estrat.", f"{cv_results['accuracy'].mean():.4f}")
	metric_cols[3].metric("Macro F1 medio CV estrat.", f"{cv_results['macro_f1'].mean():.4f}")
	if benchmark_df is not None:
		best_model = benchmark_df.sort_values("cv_macro_f1_mean", ascending=False).iloc[0]
		metric_cols[4].metric("Melhor modelo", str(best_model["model"]))
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
		power_management_counts = (
			df["wlan.fc.pwrmgt"]
			.astype("Int64")
			.astype(str)
			.value_counts()
			.sort_index()
			.rename_axis("power_management")
			.reset_index(name="frames")
		)
		power_management_chart = px.bar(
			power_management_counts,
			x="power_management",
			y="frames",
			labels={"power_management": "Power management", "frames": "Frames"},
		)
		power_management_chart.update_layout(
			margin=dict(l=0, r=0, t=10, b=0),
			xaxis_title="Power management",
			yaxis_title="Frames",
		)
		st.plotly_chart(power_management_chart, width="stretch")

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
		st.caption("A validacao cruzada usa Stratified K-Fold para preservar a distribuicao das classes em cada fold.")
		st.dataframe(cv_results, width="stretch")
		st.line_chart(cv_results.set_index("fold")[["accuracy", "macro_f1", "weighted_f1"]])
	else:
		st.info("Rode src/train_rf.py para gerar as metricas de validacao.")

	st.subheader("Validacao temporal")
	if temporal_cv_results is not None:
		st.caption("A CV temporal expanding treina com historico acumulado e testa no bloco futuro seguinte de cada dispositivo.")
		st.dataframe(temporal_cv_results, width="stretch")
		st.line_chart(temporal_cv_results.set_index("fold")[["accuracy", "macro_f1", "weighted_f1"]])
	else:
		st.info("Rode src/train_rf.py para gerar reports/temporal_cv_results.csv.")

	st.subheader("Validacao temporal sliding")
	if sliding_temporal_cv_results is not None:
		st.caption("A CV temporal sliding treina em uma janela fixa recente e testa no bloco futuro seguinte, sem acumular todo o historico.")
		st.dataframe(sliding_temporal_cv_results, width="stretch")
		st.line_chart(sliding_temporal_cv_results.set_index("fold")[["accuracy", "macro_f1", "weighted_f1"]])
	else:
		st.info("Rode src/train_rf.py para gerar reports/sliding_temporal_cv_results.csv.")

	st.subheader("Comparacao entre algoritmos")
	if benchmark_df is not None:
		display_benchmark = benchmark_df.copy()
		if "best_params" in display_benchmark.columns:
			display_benchmark["best_params"] = display_benchmark["best_params"].apply(
				lambda params: ", ".join(f"{key}={value}" for key, value in params.items()) if params else "-"
			)
		st.dataframe(display_benchmark, width="stretch")
		show_figure(FIGURES_DIR / "model_comparison_cv.png", "Comparacao entre Random Forest, KNN e SVM")
		show_figure(
			FIGURES_DIR / "model_comparison_temporal_cv.png",
			"Comparacao entre Random Forest, KNN e SVM na CV temporal",
		)
		show_figure(
			FIGURES_DIR / "model_comparison_sliding_temporal_cv.png",
			"Comparacao entre Random Forest, KNN e SVM na CV temporal sliding",
		)

		if {"model", "tuning_macro_f1", "best_params"}.issubset(benchmark_df.columns):
			st.subheader("Hiperparametros selecionados automaticamente")
			model_cols = st.columns(len(benchmark_df))
			for column, row in zip(model_cols, benchmark_df.itertuples(index=False), strict=False):
				with column:
					st.markdown(f"**{row.model}**")
					st.metric("Macro F1 medio da busca", f"{row.tuning_macro_f1:.4f}")
					st.json(row.best_params)
	else:
		st.info("Rode src/train_rf.py para gerar reports/model_benchmark.csv.")

	show_figure(FIGURES_DIR / "rf_confusion_matrix.png", "Matriz de confusao do modelo")
	show_figure(FIGURES_DIR / "rf_feature_importance.png", "Importancia das features")
	show_figure(FIGURES_DIR / "rf_cv_metrics_by_fold.png", "Evolucao das metricas por fold da CV estratificada")
	show_figure(FIGURES_DIR / "rf_temporal_cv_metrics_by_fold.png", "Evolucao das metricas por fold da CV temporal")
	show_figure(FIGURES_DIR / "rf_sliding_temporal_cv_metrics_by_fold.png", "Evolucao das metricas por fold da CV temporal sliding")

with report_tab:
	st.subheader("Relatorios")
	if report_markdowns:
		report_label = st.radio(
			"Selecione a versao do relatorio",
			options=list(report_markdowns.keys()),
			horizontal=True,
		)
		st.markdown(report_markdowns[report_label])
	else:
		st.info("Gere reports/model_report.md ou reports/academic_report.md para visualizar o relatorio.")

