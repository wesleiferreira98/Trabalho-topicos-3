from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from iot_fingerprint.model_pipeline import ModelSpec


RF_SPEC = ModelSpec(
    name="Random Forest",
    slug="rf",
    estimator_factory=lambda: RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1,
    ),
    supports_feature_importance=True,
    param_distributions={
        "n_estimators": [200, 300, 500],
        "max_depth": [None, 10, 20, 30],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2", None],
    },
    tuning_iterations=10,
)

KNN_SPEC = ModelSpec(
    name="KNN",
    slug="knn",
    estimator_factory=lambda: Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", KNeighborsClassifier(n_neighbors=5, weights="distance")),
        ]
    ),
    param_distributions={
        "model__n_neighbors": [3, 5, 7, 9, 11, 15],
        "model__weights": ["uniform", "distance"],
        "model__p": [1, 2],
    },
    tuning_iterations=8,
)

SVM_SPEC = ModelSpec(
    name="SVM",
    slug="svm",
    estimator_factory=lambda: Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", SVC(kernel="rbf", C=3.0, gamma="scale", class_weight="balanced")),
        ]
    ),
    param_distributions={
        "model__C": [0.5, 1.0, 3.0, 10.0],
        "model__gamma": ["scale", "auto", 0.01, 0.1],
        "model__kernel": ["rbf", "poly"],
    },
    tuning_iterations=8,
)

ALL_SPECS = [RF_SPEC, KNN_SPEC, SVM_SPEC]