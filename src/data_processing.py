import pandas as pd
import urllib.request
import os

# ─── Optimisation mémoire ────────────────────────────────
def optimize_memory(df):
    before = df.memory_usage(deep=True).sum() / 1024
    for col in df.columns:
        col_type = df[col].dtype
        if str(col_type).startswith('int'):
            df[col] = pd.to_numeric(df[col], downcast='integer')
        elif str(col_type).startswith('float'):
            df[col] = pd.to_numeric(df[col], downcast='float')
    after = df.memory_usage(deep=True).sum() / 1024
    print(f"Mémoire optimisée : {before:.2f} KB → {after:.2f} KB")
    return df

# This block prevents the script from running when imported by pytest
if __name__ == "__main__":
    # ─── Paths ───────────────────────────────────────────────
    BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_folder = os.path.join(BASE_DIR, "data")
    source_path = os.path.join(data_folder, "heart.csv")
    output_path = os.path.join(data_folder, "heart_balanced.csv")

    os.makedirs(data_folder, exist_ok=True)

    # ─── Téléchargement ──────────────────────────────────────
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00519/heart_failure_clinical_records_dataset.csv"
    if not os.path.exists(source_path):
        print("Téléchargement du dataset...")
        urllib.request.urlretrieve(url, source_path)
        print("Téléchargement terminé.")

    # ─── Chargement ──────────────────────────────────────────
    df = pd.read_csv(source_path)

    # NOTE: SMOTE est retiré ici pour éviter la fuite de données (data leakage).
    # Il sera appliqué UNIQUEMENT sur les données d'entraînement dans train_model.py,
    # APRÈS la séparation train/test.
    df = optimize_memory(df)

    # ─── Sauvegarde ──────────────────────────────────────────
    df.to_csv(output_path, index=False)

    if os.path.exists(output_path):
        print("--- VÉRIFICATION RÉUSSIE ---")
        print(f"Fichier sauvegardé : {output_path}")
        print(f"Taille : {os.path.getsize(output_path) / 1024:.2f} KB")
        print(f"Lignes : {len(df)} | Colonnes : {len(df.columns)}")
        print(f"Distribution DEATH_EVENT :\n{df['DEATH_EVENT'].value_counts()}")
    else:
        print("--- ERREUR : Le fichier n'a pas pu être créé ---")
