import os
import json
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
import lightgbm as lgb

# ─── Absolute paths ──────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "heart_balanced.csv")
SRC_DIR   = os.path.join(BASE_DIR, "src")

# ─── Chargement du dataset ───────────────────────────────
df = pd.read_csv(DATA_PATH)

X = df.drop("DEATH_EVENT", axis=1)
y = df["DEATH_EVENT"]

# ─── Split AVANT SMOTE (évite la fuite de données) ───────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ─── SMOTE uniquement sur les données d'entraînement ─────
smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)

print(f"Taille après SMOTE — X_train: {X_train.shape}, y_train: {y_train.shape}")
print(f"Distribution y_train: {pd.Series(y_train).value_counts().to_dict()}")
print(f"X_test (non modifié): {X_test.shape}\n")


# ─── Fonction d'évaluation ───────────────────────────────
def evaluate(name, model):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    f1  = f1_score(y_test, y_pred)
    print(f"{name:25s} | Accuracy: {acc:.3f} | ROC-AUC: {auc:.3f} | F1: {f1:.3f}")


# ─── Random Forest ───────────────────────────────────────
def train_random_forest(X_train, y_train):
    model = RandomForestClassifier(n_estimators=300, max_depth=10, random_state=42)
    model.fit(X_train, y_train)
    joblib.dump(model, os.path.join(SRC_DIR, "heart_model.pkl"))
    evaluate("Random Forest", model)
    print("  → Sauvegardé : src/heart_model.pkl")


# ─── XGBoost ─────────────────────────────────────────────
def train_xgboost(X_train, y_train):
    model = XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.05,
                          random_state=42, eval_metric="logloss")
    model.fit(X_train, y_train)
    joblib.dump(model, os.path.join(SRC_DIR, "heart_model_xgb.pkl"))
    evaluate("XGBoost", model)
    print("  → Sauvegardé : src/heart_model_xgb.pkl")


# ─── LightGBM ────────────────────────────────────────────
def train_lightgbm(X_train, y_train):
    model = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, max_depth=4,
                                num_leaves=31, random_state=42,
                                class_weight="balanced", verbose=-1)
    model.fit(X_train, y_train)
    joblib.dump(model, os.path.join(SRC_DIR, "heart_model_lgbm.pkl"))
    evaluate("LightGBM", model)
    print("  → Sauvegardé : src/heart_model_lgbm.pkl")


# ─── Logistic Regression ─────────────────────────────────
def train_logistic_regression(X_train, y_train):
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced"))
    ])
    model.fit(X_train, y_train)
    joblib.dump(model, os.path.join(SRC_DIR, "heart_model_lr.pkl"))
    evaluate("Logistic Regression", model)
    print("  → Sauvegardée : src/heart_model_lr.pkl")


# ─── Lancement ───────────────────────────────────────────
print("=== Entraînement des modèles ===\n")
train_random_forest(X_train, y_train)
train_xgboost(X_train, y_train)
train_lightgbm(X_train, y_train)
train_logistic_regression(X_train, y_train)

# ─── Sauvegarde précision réelle (APRÈS entraînement) ────
best_model = joblib.load(os.path.join(SRC_DIR, "heart_model.pkl"))
y_pred = best_model.predict(X_test)
test_accuracy = accuracy_score(y_test, y_pred)

with open(os.path.join(SRC_DIR, "metrics.json"), "w") as f:
    json.dump({"accuracy": round(test_accuracy * 100, 1)}, f)

print(f"\nPrécision test sauvegardée : {test_accuracy*100:.1f}%")
print("Tous les modèles sont entraînés et sauvegardés.")
