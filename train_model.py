# train_model.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
import joblib
import argparse

def main(csv_path, model_out="model.joblib"):
    df = pd.read_csv(csv_path)
    if df.empty:
        print("Empty CSV. Run extraction first.")
        return

    X = df.drop(columns=["label"]).values
    y_raw = df["label"].values
    le = LabelEncoder()
    y = le.fit_transform(y_raw)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    clf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    joblib.dump({"model": clf, "le": le}, model_out)
    print("Saved model to", model_out)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="landmarks csv")
    parser.add_argument("--out", default="model.joblib", help="output model file")
    args = parser.parse_args()
    main(args.csv, args.out)
