import os, json, joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import StratifiedGroupKFold, GroupShuffleSplit, cross_validate
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

os.makedirs("models", exist_ok=True)
df = pd.read_csv("data/students.csv")

# ---------- feature engineering ----------
tot = df[["video_min", "audio_min", "text_min", "practical_min"]].sum(axis=1)
df["video_share"] = df.video_min / tot
df["audio_share"] = df.audio_min / tot
df["text_share"] = df.text_min / tot
df["practical_share"] = df.practical_min / tot
df["total_weekly_min"] = tot
df["deep_indicator"] = df.revision_freq.fillna(0) + df.quiz_retries

NUM = ["video_share", "audio_share", "text_share", "practical_share",
       "total_weekly_min", "study_days_per_week", "daily_hours",
       "avg_session_min", "revision_freq", "quiz_retries", "subject_pref",
       "quiz_score", "deep_indicator"]
CAT = ["subject", "activity_type"]

# ---------- preprocessing ----------
pre = ColumnTransformer([
    ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                      ("sc", StandardScaler())]), NUM),
    ("cat", OneHotEncoder(handle_unknown="ignore"), CAT)])

MODELS = {
    "Decision Tree": DecisionTreeClassifier(max_depth=6, random_state=0),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=0),
    "KNN": KNeighborsClassifier(n_neighbors=7),
    "Logistic Regression": LogisticRegression(max_iter=2000),
}
SCORING = ["accuracy", "precision_macro", "recall_macro", "f1_macro"]

# same student never appears in both train and test
cv = StratifiedGroupKFold(5, shuffle=True, random_state=0)
groups = df.student_id

results = {}
for target in ["vark_label", "engagement_depth"]:
    X, y = df[NUM + CAT], df[target]
    rows = {}
    for name, m in MODELS.items():
        r = cross_validate(Pipeline([("pre", pre), ("m", m)]), X, y,
                           cv=cv, groups=groups, scoring=SCORING)
        rows[name] = {k: round(r[f"test_{k}"].mean(), 3) for k in SCORING}
    results[target] = rows
    top = max(rows, key=lambda n: rows[n]["f1_macro"])

    print(f"\n===== {target} (5-fold CV) =====")
    print(pd.DataFrame(rows).T.to_string())
    print("Best model:", top)

    tr, te = next(GroupShuffleSplit(1, test_size=0.2, random_state=0).split(X, y, groups))
    Xtr, Xte, ytr, yte = X.iloc[tr], X.iloc[te], y.iloc[tr], y.iloc[te]
    pipe = Pipeline([("pre", pre), ("m", MODELS[top])]).fit(Xtr, ytr)
    pred = pipe.predict(Xte)
    print(classification_report(yte, pred))
    ConfusionMatrixDisplay(confusion_matrix(yte, pred, labels=pipe.classes_),
                           display_labels=pipe.classes_).plot(cmap="Blues", xticks_rotation=30)
    plt.title(f"{target} - {top}")
    plt.tight_layout()
    plt.savefig(f"models/cm_{target}.png", dpi=120)
    plt.close()

    rf = Pipeline([("pre", pre), ("m", RandomForestClassifier(200, random_state=0))]).fit(X, y)
    names = rf.named_steps["pre"].get_feature_names_out()
    imp = pd.Series(rf.named_steps["m"].feature_importances_, index=names).sort_values()[-8:]
    imp.plot.barh(color="#2a6f97")
    plt.title(f"Top features - {target}")
    plt.tight_layout()
    plt.savefig(f"models/fi_{target}.png", dpi=120)
    plt.close()

    joblib.dump(pipe.fit(X, y), f"models/{target}.joblib")

json.dump(results, open("models/cv_results.json", "w"), indent=2)
print("\nSaved models and plots in the models/ folder")