import pandas as pd, numpy as np, json, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (confusion_matrix, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, precision_recall_curve, classification_report)

OUT = ""

train = pd.read_csv("UNSW_NB15_training-set.csv")
test = pd.read_csv("UNSW_NB15_testing-set.csv")
train.columns = train.columns.str.strip()
test.columns = test.columns.str.strip()

# ---- Data engineering ----
for df in (train, test):
    df.drop(columns=["id"], inplace=True, errors="ignore")
    df.drop_duplicates(inplace=True)

cat_cols = ["proto", "service", "state"]
num_cols = [c for c in train.columns if c not in cat_cols + ["attack_cat", "label"]]

# missing -> median(num) / mode(cat)
for c in num_cols:
    med = train[c].median()
    train[c] = train[c].fillna(med); test[c] = test[c].fillna(med)
for c in cat_cols:
    mode = train[c].mode()[0]
    train[c] = train[c].fillna(mode); test[c] = test[c].fillna(mode)

# encode categoricals (fit on train, unseen->'-1' bucket)
encoders = {}
for c in cat_cols:
    le = LabelEncoder()
    le.fit(list(train[c].astype(str).unique()) + ["__unseen__"])
    encoders[c] = le
    train[c] = le.transform(train[c].astype(str))
    test[c] = test[c].astype(str).apply(lambda v: v if v in le.classes_ else "__unseen__")
    test[c] = le.transform(test[c])

# outlier capping (winsorize at 1st/99th pct using train stats) for numeric
for c in num_cols:
    lo, hi = train[c].quantile(0.01), train[c].quantile(0.99)
    train[c] = train[c].clip(lo, hi)
    test[c] = test[c].clip(lo, hi)

X_train_raw, y_train = train[cat_cols+num_cols], train["label"].astype(int)
X_test_raw, y_test = test[cat_cols+num_cols], test["label"].astype(int)

# feature scaling (numeric only)
scaler = StandardScaler()
X_train = X_train_raw.copy(); X_test = X_test_raw.copy()
X_train[num_cols] = scaler.fit_transform(X_train_raw[num_cols])
X_test[num_cols] = scaler.transform(X_test_raw[num_cols])

# ---- Feature selection via RF importance ----
rf_fs = RandomForestClassifier(n_estimators=150, random_state=42, n_jobs=-1, class_weight="balanced")
rf_fs.fit(X_train, y_train)
importances = pd.Series(rf_fs.feature_importances_, index=X_train.columns).sort_values(ascending=False)
top_features = importances.head(15).index.tolist()
importances.to_csv(OUT+"feature_importances.csv")

plt.figure(figsize=(7,5))
importances.head(15).sort_values().plot(kind="barh", color="#3b6ea5")
plt.title("Top 15 Feature Importances (Random Forest)")
plt.tight_layout(); plt.savefig(OUT+"feature_importance.png", dpi=130); plt.close()

# ---- Final model on selected features, class-weighted (imbalance handling) ----
Xtr, Xte = X_train[top_features], X_test[top_features]
model = RandomForestClassifier(n_estimators=300, max_depth=18, random_state=42,
                                n_jobs=-1, class_weight="balanced_subsample")
model.fit(Xtr, y_train)

probs = model.predict_proba(Xte)[:,1]
preds_default = (probs >= 0.5).astype(int)

def metrics_at(th):
    p = (probs >= th).astype(int)
    cm = confusion_matrix(y_test, p)
    tn, fp, fn, tp = cm.ravel()
    return dict(threshold=th, TP=int(tp), FP=int(fp), FN=int(fn), TN=int(tn),
        precision=precision_score(y_test,p), recall=recall_score(y_test,p),
        f1=f1_score(y_test,p), FPR=fp/(fp+tn), FNR=fn/(fn+tp))

results = {
    "roc_auc": roc_auc_score(y_test, probs),
    "pr_auc": average_precision_score(y_test, probs),
    "thresholds": {str(t): metrics_at(t) for t in [0.3,0.4,0.5,0.6,0.7]},
    "top_features": top_features,
    "class_balance_train": y_train.value_counts().to_dict(),
    "class_balance_test": y_test.value_counts().to_dict(),
}
with open(OUT+"results.json","w") as f: json.dump(results, f, indent=2, default=str)

# confusion matrix plot @0.5
cm = confusion_matrix(y_test, preds_default)
plt.figure(figsize=(4.5,4))
plt.imshow(cm, cmap="Blues")
for i in range(2):
    for j in range(2):
        plt.text(j,i,cm[i,j],ha="center",va="center",
                  color="white" if cm[i,j]>cm.max()/2 else "black", fontsize=13)
plt.xticks([0,1],["Normal","Attack"]); plt.yticks([0,1],["Normal","Attack"])
plt.xlabel("Predicted"); plt.ylabel("Actual"); plt.title("Confusion Matrix (threshold=0.5)")
plt.tight_layout(); plt.savefig(OUT+"confusion_matrix.png", dpi=130); plt.close()

# PR curve
prec, rec, _ = precision_recall_curve(y_test, probs)
plt.figure(figsize=(5,4))
plt.plot(rec, prec, color="#c0392b")
plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title(f"Precision-Recall Curve (PR-AUC={results['pr_auc']:.3f})")
plt.tight_layout(); plt.savefig(OUT+"pr_curve.png", dpi=130); plt.close()

print(json.dumps(results, indent=2, default=str))
print("\nClassification report @0.5:\n", classification_report(y_test, preds_default, target_names=["Normal","Attack"]))


import joblib
joblib.dump(model, "model.pkl", compress=3)
joblib.dump(scaler, "scaler.pkl")
joblib.dump(encoders, "encoders.pkl")
joblib.dump(top_features, "top_features.pkl")