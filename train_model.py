from pathlib import Path
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

DATA = Path("data/bus_occupancy_synthetic.csv")
MODEL_DIR = Path("models")
FEATURES = ["passenger_count", "bus_capacity", "hour", "day_type",
            "previous_passenger_count", "thermal_signal"]
TARGET = "occupancy_status"

def main():
    if not DATA.exists():
        from generate_dataset import generate_dataset
        generate_dataset(output_path=str(DATA))
    df = pd.read_csv(DATA)
    X, y = df[FEATURES], df[TARGET]
    preprocessor = ColumnTransformer(
        [("day_type", OneHotEncoder(handle_unknown="ignore"), ["day_type"])],
        remainder="passthrough"
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    models = {
        "decision_tree": DecisionTreeClassifier(max_depth=6, random_state=42),
        "random_forest": RandomForestClassifier(n_estimators=150, max_depth=10,
                                               random_state=42, class_weight="balanced")
    }
    MODEL_DIR.mkdir(exist_ok=True)
    for name, clf in models.items():
        pipe = Pipeline([("preprocess", preprocessor), ("model", clf)])
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        print(f"\\n{name.upper()} accuracy: {accuracy_score(y_test, pred):.3f}")
        print(classification_report(y_test, pred, zero_division=0))
        print("Confusion matrix (label order Empty, Moderate, Full):")
        print(confusion_matrix(y_test, pred, labels=["Empty", "Moderate", "Full"]))
        joblib.dump(pipe, MODEL_DIR / f"{name}.joblib")
    print(f"Models saved to {MODEL_DIR}/")

if __name__ == "__main__":
    main()
