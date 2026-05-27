from joblib import dump
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

from iot_fingerprint.config import MODELS_DIR
from iot_fingerprint.data import load_processed_training
from iot_fingerprint.features import build_device_windows


MIN_WINDOWS_PER_DEVICE = 10


def main() -> None:
    df = load_processed_training()
    features = build_device_windows(df)

    counts = features["wlan.sa"].value_counts()
    valid_devices = counts[counts >= MIN_WINDOWS_PER_DEVICE].index
    features = features[features["wlan.sa"].isin(valid_devices)]

    x = features.drop(columns=["wlan.sa", "_window"])
    y = features["wlan.sa"]

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    model = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1,
    )
    model.fit(x_train, y_train)

    predictions = model.predict(x_test)
    print(classification_report(y_test, predictions))
    print("Matriz de confusao:")
    print(confusion_matrix(y_test, predictions, labels=model.classes_))

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    dump(model, MODELS_DIR / "random_forest_device_fingerprint.joblib")


if __name__ == "__main__":
    main()

