"""
train_model.py  –  Bot vs Human Classifier
===========================================
Trains a Random Forest (primary) and Logistic Regression (baseline),
evaluates both, and saves the best model as model.pkl.
"""

import pickle
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    classification_report
)
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# 1. LOAD PREPROCESSED DATA
# ─────────────────────────────────────────────
DATA_PATH  = "data/processed/preprocessed_data.csv"
MODEL_PATH = "app/models/model.pkl"
df = pd.read_csv(DATA_PATH)


# ─────────────────────────────────────────────
# 2. FEATURE / TARGET SPLIT
# ─────────────────────────────────────────────
TARGET = "label"   # 1 = Bot, 0 = Human

# Drop any leftover non-numeric or identifier columns if present
drop_cols = [c for c in ["login", "bio", "tag", "actor_id", "name", "email"]
             if c in df.columns]
if drop_cols:
    df.drop(columns=drop_cols, inplace=True)

X = df.drop(columns=[TARGET])
y = df[TARGET]

# ─────────────────────────────────────────────
# 3. TRAIN / TEST SPLIT  (stratified)
# ─────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


# ─────────────────────────────────────────────
# 4. DEFINE MODELS
# ─────────────────────────────────────────────

# ── (A) Random Forest  ───────────────────────
rf_model = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    min_samples_split=5,
    min_samples_leaf=2,
    class_weight="balanced",   # handles class imbalance
    random_state=42,
    n_jobs=-1
)

# ── (B) Logistic Regression baseline  ────────
lr_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("clf",    LogisticRegression(
                   max_iter=1000,
                   class_weight="balanced",
                   random_state=42,
                   solver="lbfgs"
               ))
])

# ─────────────────────────────────────────────
# 5. CROSS-VALIDATION  (5-fold, stratified)
# ─────────────────────────────────────────────
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

print("\nCross-Validation Results:\n")

for name, model in [
    ("Random Forest", rf_model),
    ("Logistic Regression", lr_pipeline)
]:
    cv_scores = cross_val_score(
        model,
        X_train,
        y_train,
        cv=cv,
        scoring="roc_auc",
        n_jobs=-1
    )

    print(f"{name}")
    print(f"ROC-AUC scores for 5 folds: {cv_scores}")
    print(f"Mean ROC-AUC: {cv_scores.mean():.4f}")
    print(f"Standard Deviation: {cv_scores.std():.4f}")
    print("-" * 50)

# ─────────────────────────────────────────────
# 6. TRAIN ON FULL TRAINING SET & EVALUATE
# ─────────────────────────────────────────────

def evaluate(name, model, X_tr, y_tr, X_te, y_te):
    model.fit(X_tr, y_tr)
    y_pred  = model.predict(X_te)
    y_proba = model.predict_proba(X_te)[:, 1]

    acc  = accuracy_score(y_te, y_pred)
    prec = precision_score(y_te, y_pred, zero_division=0)
    rec  = recall_score(y_te, y_pred, zero_division=0)
    f1   = f1_score(y_te, y_pred, zero_division=0)
    auc  = roc_auc_score(y_te, y_proba)
    cm   = confusion_matrix(y_te, y_pred)
    print(f"\n{'━'*40}")
    print(f"  {name}")
    print(f"{'━'*40}")
    print(f"  Accuracy  : {acc:.4f}")
    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}")
    print(f"  F1 Score  : {f1:.4f}")
    print(f"  ROC-AUC   : {auc:.4f}")
    print(f"\n  Confusion Matrix:")
    print(f"  {'':>10} Pred Human  Pred Bot")
    print(f"  True Human  {cm[0,0]:>8}  {cm[0,1]:>8}")
    print(f"  True Bot    {cm[1,0]:>8}  {cm[1,1]:>8}")
    print(f"\n  Classification Report:\n")
    print(classification_report(y_te, y_pred,
                                target_names=["Human (0)", "Bot (1)"],
                                zero_division=0))
    
    return {"name": name, "model": model, "auc": auc, "f1": f1}

results = []
results.append(evaluate("Random Forest",        rf_model,    X_train, y_train, X_test, y_test))
results.append(evaluate("Logistic Regression",  lr_pipeline, X_train, y_train, X_test, y_test))


# ─────────────────────────────────────────────
# 7. SELECT BEST MODEL & SAVE
# ─────────────────────────────────────────────
best = max(results, key=lambda r: r["auc"])

with open(MODEL_PATH, "wb") as f:
    pickle.dump(best["model"], f)
