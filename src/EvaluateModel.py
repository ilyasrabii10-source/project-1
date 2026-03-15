import os
import pandas as pd
import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    classification_report
)
import matplotlib.pyplot as plt

# ─── Fonctions ────────────────────────────────────────────

def evaluer_modele(nom, model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    accuracy = round(accuracy_score(y_test, y_pred) * 100, 2)
    f1       = round(f1_score(y_test, y_pred, average="weighted"), 4)
    roc_auc  = round(roc_auc_score(y_test, y_prob), 4)

    print(f"Modèle    : {nom}")
    print(f"  Accuracy  : {accuracy} %")
    print(f"  F1 Score  : {f1}")
    print(f"  ROC AUC   : {roc_auc}\n")

    return {"Modèle": nom, "Accuracy (%)": accuracy, "F1 Score": f1, "ROC AUC": roc_auc}

def load_model_and_predict(patient_data):
    """
    Loads the best model (Random Forest) and makes a prediction for a single patient.
    This function was added so tests/test_model.py works correctly!
    """
    model = joblib.load("src/heart_model.pkl")
    return model.predict(patient_data)


# ─── Exécution du Script ──────────────────────────────────
if __name__ == "__main__":
    
    # Creates the 'models' directory if it doesn't exist to prevent FileNotFoundError
    os.makedirs("models", exist_ok=True)

    # ─── Chargement des données ───────────────────────────────
    df = pd.read_csv("data/nouvelle_dataset_equilibree.csv")

    X = df.drop("DEATH_EVENT", axis=1)
    y = df["DEATH_EVENT"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # ─── Chargement des modèles ✅ noms corrects ──────────────
    model_rf  = joblib.load("src/heart_model.pkl")
    model_xgb = joblib.load("src/heart_model_xgb.pkl")
    model_lgb = joblib.load("src/heart_model_lgbm.pkl")
    model_lr  = joblib.load("src/heart_model_lr.pkl")

    # ─── Évaluation de tous les modèles ──────────────────────
    resultats = []
    resultats.append(evaluer_modele("Random Forest",      model_rf, X_test, y_test))
    resultats.append(evaluer_modele("XGBoost",            model_xgb, X_test, y_test))
    resultats.append(evaluer_modele("LightGBM",           model_lgb, X_test, y_test))
    resultats.append(evaluer_modele("Logistic Regression",model_lr, X_test, y_test))

    # ─── Tableau récapitulatif ────────────────────────────────
    df_resultats = pd.DataFrame(resultats)
    print("=" * 52)
    print("         COMPARAISON DES MODÈLES")
    print("=" * 52)
    print(df_resultats.to_string(index=False))
    print("=" * 52)

    meilleur = df_resultats.loc[df_resultats["ROC AUC"].idxmax(), "Modèle"]
    print(f"\n🏆 Meilleur modèle (ROC AUC) : {meilleur}")

    # ─── Rapport détaillé par modèle ─────────────────────────
    print("\n" + "=" * 52)
    print("           RAPPORTS DÉTAILLÉS")
    print("=" * 52)

    modeles = {
        "Random Forest":       model_rf,
        "XGBoost":             model_xgb,
        "LightGBM":            model_lgb,
        "Logistic Regression": model_lr,
    }

    for nom, model in modeles.items():
        y_pred = model.predict(X_test)
        print(f"\n🔹 {nom}")
        print(classification_report(y_test, y_pred, target_names=["Survie", "Décès"]))

    # ─── Visualisation comparative ────────────────────────────
    metrics     = ["Accuracy (%)", "F1 Score", "ROC AUC"]
    model_names = df_resultats["Modèle"].tolist()
    x           = np.arange(len(metrics))
    width       = 0.2
    colors      = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]

    fig, ax = plt.subplots(figsize=(11, 6))

    for i, (nom, color) in enumerate(zip(model_names, colors)):
        row  = df_resultats[df_resultats["Modèle"] == nom].iloc[0]
        vals = [row["Accuracy (%)"] / 100, row["F1 Score"], row["ROC AUC"]]
        bars = ax.bar(x + i * width, vals, width, label=nom, color=color, alpha=0.85)
        for bar, val in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f"{val:.3f}",
                ha="center", va="bottom", fontsize=8
            )

    ax.set_xlabel("Métriques")
    ax.set_ylabel("Score")
    ax.set_title("Comparaison des modèles — Heart Failure Prediction")
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(["Accuracy", "F1 Score", "ROC AUC"])
    ax.set_ylim(0, 1.15)
    ax.legend(loc="lower right")
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig("models/comparaison_modeles.png", dpi=150)
    # plt.show() # Optional: Commented out so it doesn't pause scripts on servers

    print("\n✅ Graphique sauvegardé : models/comparaison_modeles.png")
